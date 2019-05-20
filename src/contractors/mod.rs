use crate::SizedContraction;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::HashMap;

mod singleton_contractors;
use singleton_contractors::{
    Diagonalization, DiagonalizationAndSummation, Identity, Permutation, PermutationAndSummation,
    Summation,
};

mod pair_contractors;
pub use pair_contractors::{HadamardProductGeneral, TensordotFixedPosition, TensordotGeneral, ScalarMatrixProductGeneral};

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


pub struct SingletonContraction<A> {
    op: Box<dyn SingletonContractor<A>>,
}

impl<A> SingletonContraction<A> {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 1);
        let output_indices = &sc.contraction.output_indices;
        let input_indices = &sc.contraction.operand_indices[0];
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

        match (num_summed_axes, num_diagonalized_axes, num_reordered_axes) {
            (0, 0, 0) => {
                let identity = Identity::new();
                SingletonContraction {
                    op: Box::new(identity),
                }
            }
            (0, 0, _) => {
                let permutation = Permutation::new(sc);
                SingletonContraction {
                    op: Box::new(permutation),
                }
            }
            (_, 0, 0) => {
                let summation = Summation::new(sc);
                SingletonContraction {
                    op: Box::new(summation),
                }
            }
            (0, _, _) => {
                let diagonalization = Diagonalization::new(sc);
                SingletonContraction {
                    op: Box::new(diagonalization),
                }
            }
            (_, 0, _) => {
                let permutation_and_summation = PermutationAndSummation::new(sc);
                SingletonContraction {
                    op: Box::new(permutation_and_summation),
                }
            }
            (_, _, _) => {
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
