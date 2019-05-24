use crate::optimizers::{
    generate_optimized_path, EinsumPath, FirstStep, IntermediateStep, OperandNumPair,
    OptimizationMethod,
};
use crate::{ArrayLike, SizedContraction};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::HashSet;

mod singleton_contractors;
use singleton_contractors::{
    Diagonalization, DiagonalizationAndSummation, Identity, Permutation, PermutationAndSummation,
    Summation,
};

mod pair_contractors;
pub use pair_contractors::TensordotGeneral;
use pair_contractors::{
    BroadcastProductGeneral, HadamardProduct, HadamardProductGeneral, MatrixScalarProduct,
    MatrixScalarProductGeneral, ScalarMatrixProduct, ScalarMatrixProductGeneral,
    StackedTensordotGeneral, TensordotFixedPosition,
};

mod strategies;
use strategies::{PairMethod, PairSummary, SingletonMethod, SingletonSummary};

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

pub trait PathContractor<A> {
    fn contract_path(&self, operands: &[&dyn ArrayLike<A>]) -> ArrayD<A>
    where
        A: Clone + LinalgScalar;
}

pub struct SingletonContraction<A> {
    op: Box<dyn SingletonContractor<A>>,
}

impl<A> SingletonContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        let singleton_summary = SingletonSummary::new(&sc);

        match singleton_summary.get_strategy() {
            SingletonMethod::Identity => SingletonContraction {
                op: Box::new(Identity::new()),
            },
            SingletonMethod::Permutation => SingletonContraction {
                op: Box::new(Permutation::new(sc)),
            },
            SingletonMethod::Summation => SingletonContraction {
                op: Box::new(Summation::new(sc)),
            },
            SingletonMethod::Diagonalization => SingletonContraction {
                op: Box::new(Diagonalization::new(sc)),
            },
            SingletonMethod::PermutationAndSummation => SingletonContraction {
                op: Box::new(PermutationAndSummation::new(sc)),
            },
            SingletonMethod::DiagonalizationAndSummation => SingletonContraction {
                op: Box::new(DiagonalizationAndSummation::new(sc)),
            },
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

        let SimplificationMethodAndOutput {
            method: lhs_simplification,
            new_indices: new_lhs_indices,
        } = SimplificationMethodAndOutput::from_indices_and_sizes(
            &lhs_indices,
            &rhs_indices,
            &output_indices,
            sc,
        );
        let SimplificationMethodAndOutput {
            method: rhs_simplification,
            new_indices: new_rhs_indices,
        } = SimplificationMethodAndOutput::from_indices_and_sizes(
            &rhs_indices,
            &lhs_indices,
            &output_indices,
            sc,
        );

        let reduced_sc = sc
            .subset(&[new_lhs_indices, new_rhs_indices], &output_indices)
            .unwrap();

        let pair_summary = PairSummary::new(&reduced_sc);
        let pair_strategy = pair_summary.get_strategy();

        let op: Box<dyn PairContractor<A>> = match pair_strategy {
            PairMethod::HadamardProduct => {
                // Never gets returned in current implementation
                Box::new(HadamardProduct::new(&reduced_sc))
            }
            PairMethod::HadamardProductGeneral => {
                Box::new(HadamardProductGeneral::new(&reduced_sc))
            }
            PairMethod::ScalarMatrixProduct => {
                // Never gets returned in current implementation
                Box::new(ScalarMatrixProduct::new(&reduced_sc))
            }
            PairMethod::ScalarMatrixProductGeneral => {
                Box::new(ScalarMatrixProductGeneral::new(&reduced_sc))
            }
            PairMethod::MatrixScalarProduct => {
                // Never gets returned in current implementation
                Box::new(MatrixScalarProduct::new(&reduced_sc))
            }
            PairMethod::MatrixScalarProductGeneral => {
                Box::new(MatrixScalarProductGeneral::new(&reduced_sc))
            }
            PairMethod::TensordotFixedPosition => {
                // Never gets returned in current implementation
                Box::new(TensordotFixedPosition::new(&reduced_sc))
            }
            PairMethod::TensordotGeneral => Box::new(TensordotGeneral::new(&reduced_sc)),
            PairMethod::StackedTensordotGeneral => {
                Box::new(StackedTensordotGeneral::new(&reduced_sc))
            }
            PairMethod::BroadcastProductGeneral => {
                // Never gets returned in current implementation
                Box::new(BroadcastProductGeneral::new(&reduced_sc))
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

pub enum PathContractionSteps<A> {
    SingletonContraction(SingletonContraction<A>),
    PairContractions(Vec<PairContraction<A>>),
}

pub struct PathContraction<A> {
    pub path: EinsumPath,
    pub steps: PathContractionSteps<A>,
}

impl<A> PathContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        let path = generate_optimized_path(&sc, OptimizationMethod::Naive);

        PathContraction::from_path(&path)
    }

    pub fn from_path(path: &EinsumPath) -> Self {
        let EinsumPath {
            first_step:
                FirstStep {
                    ref sized_contraction,
                    ref operand_nums,
                },
            ref remaining_steps,
        } = path;

        match operand_nums {
            None => PathContraction {
                path: path.clone(),
                steps: PathContractionSteps::SingletonContraction(SingletonContraction::new(
                    sized_contraction,
                )),
            },
            Some(OperandNumPair { .. }) => {
                let mut steps = Vec::new();
                steps.push(PairContraction::new(sized_contraction));

                for step in remaining_steps.iter() {
                    steps.push(PairContraction::new(&step.sized_contraction));
                }

                PathContraction {
                    path: path.clone(),
                    steps: PathContractionSteps::PairContractions(steps),
                }
            }
        }
    }
}

impl<A> PathContractor<A> for PathContraction<A> {
    fn contract_path(&self, operands: &[&dyn ArrayLike<A>]) -> ArrayD<A>
    where
        A: Clone + LinalgScalar,
    {
        let EinsumPath {
            first_step:
                FirstStep {
                    ref operand_nums,
                    ..
                },
            ref remaining_steps,
        } = &self.path;

        match &self.steps {
            PathContractionSteps::SingletonContraction(c) => match operand_nums {
                None => c.contract_singleton(&operands[0].into_dyn_view()),
                Some(_) => panic!(),
            },
            PathContractionSteps::PairContractions(steps) => {
                let mut result = match operand_nums {
                    None => panic!(),
                    Some(OperandNumPair { lhs, rhs }) => steps[0].contract_pair(
                        &(operands[*lhs].into_dyn_view()),
                        &(operands[*rhs].into_dyn_view()),
                    ),
                };
                for (operand_step, pair_contraction_step) in
                    remaining_steps.iter().zip(steps[1..].iter())
                {
                    let IntermediateStep { ref rhs_num, .. } = operand_step;
                    result = pair_contraction_step
                        .contract_pair(&result.view(), &(operands[*rhs_num].into_dyn_view()));
                }
                result
            }
        }
    }
}
