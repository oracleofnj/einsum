use crate::classifiers::generate_classified_pair_contraction;
use crate::validation::OutputSize;
use crate::{Contraction, SizedContraction};
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

enum SingletonSimplificationMethod<A> {
    Contraction(Box<dyn SingletonContractor<A>>),
    View(Box<dyn SingletonViewer<A>>),
}

struct SimplificationMethodAndOutput<A> {
    method: SingletonSimplificationMethod<A>,
    new_indices: Vec<char>,
}

impl<A> SimplificationMethodAndOutput<A> {
    fn from_indices_and_sizes(
        this_input_indices: &[char],
        other_input_indices: &[char],
        output_indices: &[char],
        output_size: &OutputSize,
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
        let new_indices: Vec<char> = desired_uniques.iter().cloned().collect();

        let simplification_sc = SizedContraction {
            contraction: Contraction {
                operand_indices: vec![this_input_indices.to_vec()],
                summation_indices: this_input_uniques
                    .difference(&desired_uniques)
                    .cloned()
                    .collect(),
                output_indices: new_indices.clone(),
            },
            output_size: output_size.clone(),
        };
        println!("simplification: {:?}", &simplification_sc);

        let singleton_summary = SingletonSummary::new(&simplification_sc);

        let method = match singleton_summary.get_strategy() {
            SingletonMethod::Identity => {
                let identity = Identity::new();
                SingletonSimplificationMethod::View(Box::new(identity))
            }
            SingletonMethod::Permutation => {
                let permutation = Permutation::new(&simplification_sc);
                SingletonSimplificationMethod::View(Box::new(permutation))
            }
            SingletonMethod::Summation => {
                let summation = Summation::new(&simplification_sc);
                SingletonSimplificationMethod::Contraction(Box::new(summation))
            }
            SingletonMethod::Diagonalization => {
                let diagonalization = Diagonalization::new(&simplification_sc);
                SingletonSimplificationMethod::Contraction(Box::new(diagonalization))
            }
            SingletonMethod::PermutationAndSummation => {
                let permutation_and_summation = PermutationAndSummation::new(&simplification_sc);
                SingletonSimplificationMethod::Contraction(Box::new(permutation_and_summation))
            }
            SingletonMethod::DiagonalizationAndSummation => {
                let diagonalization_and_summation =
                    DiagonalizationAndSummation::new(&simplification_sc);
                SingletonSimplificationMethod::Contraction(Box::new(diagonalization_and_summation))
            }
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

        println!("lhs:");
        let lhs_simplification_and_output = SimplificationMethodAndOutput::from_indices_and_sizes(
            &lhs_indices,
            &rhs_indices,
            &output_indices,
            &sc.output_size,
        );
        println!("rhs:");
        let rhs_simplification_and_output = SimplificationMethodAndOutput::from_indices_and_sizes(
            &rhs_indices,
            &lhs_indices,
            &output_indices,
            &sc.output_size,
        );
        let lhs_simplification = lhs_simplification_and_output.method;
        let new_lhs_indices = lhs_simplification_and_output.new_indices;
        let rhs_simplification = rhs_simplification_and_output.method;
        let new_rhs_indices = rhs_simplification_and_output.new_indices;

        let mut both_indices = new_lhs_indices.clone();
        both_indices.append(&mut new_rhs_indices.clone());
        let mut summation_indices: Vec<char> = both_indices
            .iter()
            .filter(|&&c| output_indices.iter().position(|&x| x == c).is_none())
            .cloned()
            .collect();
        summation_indices.sort();
        summation_indices.dedup();

        let reduced_sc = SizedContraction {
            contraction: Contraction {
                operand_indices: vec![new_lhs_indices, new_rhs_indices],
                summation_indices,
                output_indices: output_indices.clone(),
            },
            output_size: sc.output_size.clone(),
        };
        println!("reduced: {:?}", &reduced_sc);
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
            // (_, _, 0, _, _, _, _) => {
            //     let contractor = BroadcastProductGeneral::new(&sized_contraction);
            //     contractor.contract_pair(lhs, rhs)
            // }
            // (_, _, 0, _, _, false, _) => {
            //     let contractor = BroadcastProductGeneral::new(&sized_contraction);
            //     contractor.contract_pair(lhs, rhs)
            // }
            // (_, _, 0, _, _, _, false) => {
            //     let contractor = BroadcastProductGeneral::new(&sized_contraction);
            //     contractor.contract_pair(lhs, rhs)
            // }
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
        let contracted_lhs;
        let reduced_lhs = match &self.lhs_simplification {
            SingletonSimplificationMethod::Contraction(c) => {
                contracted_lhs = c.contract_singleton(lhs);
                contracted_lhs.view()
            }
            SingletonSimplificationMethod::View(v) => v.view_singleton(lhs),
        };
        let contracted_rhs;
        let reduced_rhs = match &self.rhs_simplification {
            SingletonSimplificationMethod::Contraction(c) => {
                contracted_rhs = c.contract_singleton(rhs);
                contracted_rhs.view()
            }
            SingletonSimplificationMethod::View(v) => v.view_singleton(rhs),
        };
        self.op.contract_pair(&reduced_lhs, &reduced_rhs)
    }
}
