use crate::classifiers::generate_classified_pair_contraction;
use crate::SizedContraction;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::{HashMap, HashSet};

mod singleton_contractors;
use singleton_contractors::{
    Diagonalization, DiagonalizationAndSummation, Identity, Permutation, PermutationAndSummation,
    Summation,
};

mod pair_contractors;
pub use pair_contractors::{
    BroadcastProductGeneral, HadamardProductGeneral, MatrixScalarProductGeneral,
    ScalarMatrixProductGeneral, StackedTensordotGeneral, TensordotFixedPosition, TensordotGeneral,
};

pub trait SingletonViewer<A> {
    fn view_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayViewD<'b, A>
    where
        'a: 'b,
        A: Clone + LinalgScalar;
}

pub trait SingletonContractor<A> {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar;
}

pub trait PairContractor<A> {
    fn contract_pair<'a, 'b, 'c, 'd>(
        &self,
        lhs: &'b ArrayViewD<'a, A>,
        rhs: &'d ArrayViewD<'c, A>,
    ) -> ArrayD<A>
    where
        'a: 'b,
        'c: 'd,
        A: Clone + LinalgScalar;

    fn contract_and_assign_pair<'a, 'b, 'c, 'd, 'e, 'f>(
        &self,
        lhs: &'b ArrayViewD<'a, A>,
        rhs: &'d ArrayViewD<'c, A>,
        out: &'f mut ArrayViewMutD<'e, A>,
    ) where
        'a: 'b,
        'c: 'd,
        'e: 'f,
        A: Clone + LinalgScalar,
    {
        let result = self.contract_pair(lhs, rhs);
        out.assign(&result);
    }
}

struct SingletonSummary {
    num_summed_axes: usize,
    num_diagonalized_axes: usize,
    num_reordered_axes: usize,
}

enum SingletonMethod {
    Identity,
    Permutation,
    Summation,
    Diagonalization,
    PermutationAndSummation,
    DiagonalizationAndSummation,
}

impl SingletonSummary {
    fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 1);
        let output_indices = &sc.contraction.output_indices;
        let input_indices = &sc.contraction.operand_indices[0];

        SingletonSummary::from_indices(&input_indices, &output_indices)
    }

    fn from_indices(input_indices: &[char], output_indices: &[char]) -> Self {
        let mut input_counts = HashMap::new();
        for &c in input_indices.iter() {
            *input_counts.entry(c).or_insert(0) += 1;
        }
        let num_summed_axes = input_counts.len() - output_indices.len();
        let num_diagonalized_axes = input_counts.iter().filter(|(_, &v)| v > 1).count();
        let num_reordered_axes = output_indices
            .iter()
            .zip(input_indices.iter())
            .filter(|(&output_char, &input_char)| output_char != input_char)
            .count();

        SingletonSummary {
            num_summed_axes,
            num_diagonalized_axes,
            num_reordered_axes,
        }
    }

    fn get_strategy(&self) -> SingletonMethod {
        match (
            self.num_summed_axes,
            self.num_diagonalized_axes,
            self.num_reordered_axes,
        ) {
            (0, 0, 0) => SingletonMethod::Identity,
            (0, 0, _) => SingletonMethod::Permutation,
            (_, 0, 0) => SingletonMethod::Summation,
            (0, _, _) => SingletonMethod::Diagonalization,
            (_, 0, _) => SingletonMethod::PermutationAndSummation,
            (_, _, _) => SingletonMethod::DiagonalizationAndSummation,
        }
    }
}

pub struct SingletonContraction<A> {
    op: Box<dyn SingletonContractor<A>>,
}

impl<A> SingletonContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        let singleton_summary = SingletonSummary::new(&sc);

        match singleton_summary.get_strategy() {
            SingletonMethod::Identity => {
                let identity = Identity::new();
                SingletonContraction {
                    op: Box::new(identity),
                }
            }
            SingletonMethod::Permutation => {
                let permutation = Permutation::new(sc);
                SingletonContraction {
                    op: Box::new(permutation),
                }
            }
            SingletonMethod::Summation => {
                let summation = Summation::new(sc);
                SingletonContraction {
                    op: Box::new(summation),
                }
            }
            SingletonMethod::Diagonalization => {
                let diagonalization = Diagonalization::new(sc);
                SingletonContraction {
                    op: Box::new(diagonalization),
                }
            }
            SingletonMethod::PermutationAndSummation => {
                let permutation_and_summation = PermutationAndSummation::new(sc);
                SingletonContraction {
                    op: Box::new(permutation_and_summation),
                }
            }
            SingletonMethod::DiagonalizationAndSummation => {
                let diagonalization_and_summation = DiagonalizationAndSummation::new(sc);
                SingletonContraction {
                    op: Box::new(diagonalization_and_summation),
                }
            }
        }
    }
}

impl<A> SingletonContractor<A> for SingletonContraction<A> {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        self.op.contract_singleton(tensor)
    }
}

type SingletonSimplificationMethod<A> = Option<Box<dyn SingletonContractor<A>>>;

struct SimplificationMethodAndOutput<A> {
    method: SingletonSimplificationMethod<A>,
    new_indices: Vec<char>,
}

impl<A> SimplificationMethodAndOutput<A> {
    fn from_indices_and_sizes(
        this_input_indices: &[char],
        other_input_indices: &[char],
        output_indices: &[char],
        orig_contraction: &SizedContraction,
    ) -> Self {
        let this_input_uniques: HashSet<char> = this_input_indices.iter().cloned().collect();
        let other_input_uniques: HashSet<char> = other_input_indices.iter().cloned().collect();
        let output_uniques: HashSet<char> = output_indices.iter().cloned().collect();

        let other_and_output: HashSet<char> = other_input_uniques
            .union(&output_uniques)
            .cloned()
            .collect();
        let desired_uniques: HashSet<char> = this_input_uniques
            .intersection(&other_and_output)
            .cloned()
            .collect();
        let simplified_indices: Vec<char> = desired_uniques.iter().cloned().collect();

        let simplification_sc = orig_contraction
            .subset(&[this_input_indices.to_vec()], &simplified_indices)
            .unwrap();

        let singleton_summary = SingletonSummary::new(&simplification_sc);

        let method: Option<Box<dyn SingletonContractor<A>>> = match singleton_summary.get_strategy()
        {
            SingletonMethod::Identity => None,
            SingletonMethod::Permutation => None,
            SingletonMethod::Summation => {
                let summation = Summation::new(&simplification_sc);
                Some(Box::new(summation))
            }
            SingletonMethod::Diagonalization => {
                let diagonalization = Diagonalization::new(&simplification_sc);
                Some(Box::new(diagonalization))
            }
            SingletonMethod::PermutationAndSummation => {
                let permutation_and_summation = PermutationAndSummation::new(&simplification_sc);
                Some(Box::new(permutation_and_summation))
            }
            SingletonMethod::DiagonalizationAndSummation => {
                let diagonalization_and_summation =
                    DiagonalizationAndSummation::new(&simplification_sc);
                Some(Box::new(diagonalization_and_summation))
            }
        };
        let new_indices = if method.is_some() {
            simplified_indices
        } else {
            this_input_indices.to_vec()
        };

        SimplificationMethodAndOutput {
            method,
            new_indices,
        }
    }
}

pub struct PairContraction<A> {
    lhs_simplification: SingletonSimplificationMethod<A>,
    rhs_simplification: SingletonSimplificationMethod<A>,
    op: Box<dyn PairContractor<A>>,
}

impl<A> PairContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;

        let lhs_simplification_and_output = SimplificationMethodAndOutput::from_indices_and_sizes(
            &lhs_indices,
            &rhs_indices,
            &output_indices,
            sc,
        );
        let rhs_simplification_and_output = SimplificationMethodAndOutput::from_indices_and_sizes(
            &rhs_indices,
            &lhs_indices,
            &output_indices,
            sc,
        );
        let lhs_simplification = lhs_simplification_and_output.method;
        let new_lhs_indices = lhs_simplification_and_output.new_indices;
        let rhs_simplification = rhs_simplification_and_output.method;
        let new_rhs_indices = rhs_simplification_and_output.new_indices;

        let reduced_sc = sc
            .subset(&[new_lhs_indices, new_rhs_indices], &output_indices)
            .unwrap();

        let cpc = generate_classified_pair_contraction(&reduced_sc);

        let op: Box<dyn PairContractor<A>> = match (
            cpc.stack_indices.len(),
            cpc.outer_indices.len(),
            cpc.contracted_indices.len(),
            cpc.lhs_indices.len(),
            cpc.rhs_indices.len(),
            cpc.lhs_indices.len() > cpc.stack_indices.len(),
            cpc.rhs_indices.len() > cpc.stack_indices.len(),
        ) {
            (_, 0, 0, _, _, _, _) => {
                let hadamarder = HadamardProductGeneral::new(&reduced_sc);
                Box::new(hadamarder)
            }
            (0, _, 0, 0, _, _, _) => {
                let scalermatrixer = ScalarMatrixProductGeneral::new(&reduced_sc);
                Box::new(scalermatrixer)
            }
            (0, _, 0, _, 0, _, _) => {
                let matrixscalerer = MatrixScalarProductGeneral::new(&reduced_sc);
                Box::new(matrixscalerer)
            }
            (0, _, _, _, _, _, _) => {
                let tensordotter = TensordotGeneral::new(&reduced_sc);
                Box::new(tensordotter)
            }
            (_, _, _, _, _, _, _) => {
                let contractor = StackedTensordotGeneral::new(&reduced_sc);
                Box::new(contractor)
            }
        };
        PairContraction {
            lhs_simplification,
            rhs_simplification,
            op,
        }
    }
}

impl<A> PairContractor<A> for PairContraction<A> {
    fn contract_pair<'a, 'b, 'c, 'd>(
        &self,
        lhs: &'b ArrayViewD<'a, A>,
        rhs: &'d ArrayViewD<'c, A>,
    ) -> ArrayD<A>
    where
        'a: 'b,
        'c: 'd,
        A: Clone + LinalgScalar,
    {
        match (&self.lhs_simplification, &self.rhs_simplification) {
            (None, None) => self.op.contract_pair(lhs, rhs),
            (Some(lhs_contraction), None) => self
                .op
                .contract_pair(&lhs_contraction.contract_singleton(lhs).view(), rhs),
            (None, Some(rhs_contraction)) => self
                .op
                .contract_pair(lhs, &rhs_contraction.contract_singleton(rhs).view()),
            (Some(lhs_contraction), Some(rhs_contraction)) => self.op.contract_pair(
                &lhs_contraction.contract_singleton(lhs).view(),
                &rhs_contraction.contract_singleton(rhs).view(),
            ),
        }
    }
}
