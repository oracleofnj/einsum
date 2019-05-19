use ndarray::prelude::*;
use ndarray::LinalgScalar;

use super::{PairContractor, SingletonContractor, SingletonViewer};
use crate::{Contraction, SizedContraction};

#[derive(Clone, Debug)]
pub struct TensordotFixedPosition {
    len_contracted_lhs: usize,
    len_uncontracted_lhs: usize,
    len_contracted_rhs: usize,
    len_uncontracted_rhs: usize,
    output_shape: Vec<usize>,
}

impl TensordotFixedPosition {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        // Returns an n-dimensional array where n = |D| + |E| - 2 * last_n.
        let twice_num_contracted_axes =
            lhs_indices.len() + rhs_indices.len() - output_indices.len();
        assert_eq!(twice_num_contracted_axes % 2, 0);
        let num_contracted_axes = twice_num_contracted_axes / 2;
        // TODO: Add an assert! that they have the same indices

        let lhs_shape: Vec<usize> = lhs_indices.iter().map(|c| sc.output_size[c]).collect();
        let rhs_shape: Vec<usize> = rhs_indices.iter().map(|c| sc.output_size[c]).collect();

        TensordotFixedPosition::from_shapes_and_number_of_contracted_axes(
            &lhs_shape,
            &rhs_shape,
            num_contracted_axes,
        )
    }

    pub fn from_shapes_and_number_of_contracted_axes(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        num_contracted_axes: usize,
    ) -> Self {
        let mut len_uncontracted_lhs = 1;
        let mut len_uncontracted_rhs = 1;
        let mut len_contracted_lhs = 1;
        let mut len_contracted_rhs = 1;
        let mut output_shape = Vec::new();

        let num_axes_lhs = lhs_shape.len();
        for (axis, &axis_length) in lhs_shape.iter().enumerate() {
            if axis < (num_axes_lhs - num_contracted_axes) {
                len_uncontracted_lhs *= axis_length;
                output_shape.push(axis_length);
            } else {
                len_contracted_lhs *= axis_length;
            }
        }

        for (axis, &axis_length) in rhs_shape.iter().enumerate() {
            if axis < num_contracted_axes {
                len_contracted_rhs *= axis_length;
            } else {
                len_uncontracted_rhs *= axis_length;
                output_shape.push(axis_length);
            }
        }

        TensordotFixedPosition {
            len_contracted_lhs,
            len_uncontracted_lhs,
            len_contracted_rhs,
            len_uncontracted_rhs,
            output_shape,
        }
    }
}

impl<A> PairContractor<A> for TensordotFixedPosition {
    fn contract_pair<'a, 'b>(
        &self,
        lhs: &'b ArrayViewD<'a, A>,
        rhs: &'b ArrayViewD<'a, A>,
    ) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        let lhs_array;
        let lhs_view = if lhs.is_standard_layout() {
            lhs.view()
                .into_shape((self.len_uncontracted_lhs, self.len_contracted_lhs))
                .unwrap()
        } else {
            lhs_array = Array::from_shape_vec(
                [self.len_uncontracted_lhs, self.len_contracted_lhs],
                lhs.iter().cloned().collect(),
            )
            .unwrap();
            lhs_array.view()
        };

        let rhs_array;
        let rhs_view = if rhs.is_standard_layout() {
            rhs.view()
                .into_shape((self.len_contracted_rhs, self.len_uncontracted_rhs))
                .unwrap()
        } else {
            rhs_array = Array::from_shape_vec(
                [self.len_contracted_rhs, self.len_uncontracted_rhs],
                rhs.iter().cloned().collect(),
            )
            .unwrap();
            rhs_array.view()
        };

        lhs_view
            .dot(&rhs_view)
            .into_shape(IxDyn(&self.output_shape))
            .unwrap()
    }
}
