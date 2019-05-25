use std::fmt::Debug;
use crate::optimizers::{
    generate_optimized_order, ContractionOrder, OperandNumber, OptimizationMethod,
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

pub trait SingletonViewer<A>: Debug {
    fn view_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayViewD<'b, A>
    where
        'a: 'b,
        A: Clone + LinalgScalar;
}

pub trait SingletonContractor<A>: Debug {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar;
}

pub trait PairContractor<A>: Debug {
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

pub trait PathContractor<A>: Debug {
    fn contract_operands(&self, operands: &[&dyn ArrayLike<A>]) -> ArrayD<A>
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

impl<A> Debug for SingletonContraction<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "op: {:?}", self.op)
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

impl<A> Debug for SimplificationMethodAndOutput<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.method {
            Some(_) => write!(f, "method: Some({:?}), new_indices: {:?}", &self.method, &self.new_indices),
            None => write!(f, "None")
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

impl<A> Debug for PairContraction<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "lhs_simplification: {:?}, rhs_simplification: {:?}, op: {:?}", self.lhs_simplification, self.rhs_simplification, self.op)
    }
}

pub enum PathContractionSteps<A> {
    SingletonContraction(SingletonContraction<A>),
    PairContractions(Vec<PairContraction<A>>),
}

pub struct PathContraction<A> {
    pub contraction_order: ContractionOrder,
    pub steps: PathContractionSteps<A>,
}

impl<A> PathContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        let contraction_order = generate_optimized_order(&sc, OptimizationMethod::Naive);

        PathContraction::from_path(&contraction_order)
    }

    pub fn from_path(contraction_order: &ContractionOrder) -> Self {
        match contraction_order {
            ContractionOrder::Singleton(sized_contraction) => PathContraction {
                contraction_order: contraction_order.clone(),
                steps: PathContractionSteps::SingletonContraction(SingletonContraction::new(
                    sized_contraction,
                )),
            },
            ContractionOrder::Pairs(order_steps) => {
                let mut steps = Vec::new();

                for step in order_steps.iter() {
                    steps.push(PairContraction::new(&step.sized_contraction));
                }

                PathContraction {
                    contraction_order: contraction_order.clone(),
                    steps: PathContractionSteps::PairContractions(steps),
                }
            }
        }
    }
}

impl<A> PathContractor<A> for PathContraction<A> {
    fn contract_operands(&self, operands: &[&dyn ArrayLike<A>]) -> ArrayD<A>
    where
        A: Clone + LinalgScalar,
    {
        // Uncomment for help debugging
        // println!("{:?}", self);
        match (&self.steps, &self.contraction_order) {
            (PathContractionSteps::SingletonContraction(c), ContractionOrder::Singleton(_)) => {
                c.contract_singleton(&operands[0].into_dyn_view())
            }
            (
                PathContractionSteps::PairContractions(steps),
                ContractionOrder::Pairs(order_steps),
            ) => {
                let mut intermediate_results: Vec<ArrayD<A>> = Vec::new();
                for (step, order_step) in steps.iter().zip(order_steps.iter()) {
                    let lhs = match order_step.operand_nums.lhs {
                        OperandNumber::Input(pos) => operands[pos].into_dyn_view(),
                        OperandNumber::IntermediateResult(pos) => intermediate_results[pos].view(),
                    };
                    let rhs = match order_step.operand_nums.rhs {
                        OperandNumber::Input(pos) => operands[pos].into_dyn_view(),
                        OperandNumber::IntermediateResult(pos) => intermediate_results[pos].view(),
                    };
                    let intermediate_result = step.contract_pair(&lhs, &rhs);
                    // let lhs = match order_step.
                    intermediate_results.push(intermediate_result);
                }
                intermediate_results.pop().unwrap()
            }
            _ => panic!(), // steps and contraction_order don't match
        }
    }
}

impl<A> Debug for PathContraction<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.steps {
            PathContractionSteps::SingletonContraction(step) => write!(f, "only_step: {:?}", step),
            PathContractionSteps::PairContractions(steps) => write!(f, "steps: {:?}", steps)
        }
    }
}
