use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::{HashMap, HashSet};

use super::{PairContractor, Permutation, SingletonContractor, SingletonViewer};
use crate::{Contraction, SizedContraction};

fn find_outputs_in_inputs_unique(output_indices: &[char], input_indices: &[char]) -> Vec<usize> {
    output_indices
        .iter()
        .map(|&output_char| {
            let input_pos = input_indices
                .iter()
                .position(|&input_char| input_char == output_char)
                .unwrap();
            assert!(input_indices
                .iter()
                .skip(input_pos + 1)
                .position(|&input_char| input_char == output_char)
                .is_none());
            input_pos
        })
        .collect()
}

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

#[derive(Clone, Debug)]
pub struct TensordotGeneral {
    lhs_permutation: Permutation,
    rhs_permutation: Permutation,
    tensordot_fixed_position: TensordotFixedPosition,
}

impl TensordotGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;

        let lhs_shape: Vec<usize> = lhs_indices.iter().map(|c| sc.output_size[c]).collect();
        let rhs_shape: Vec<usize> = rhs_indices.iter().map(|c| sc.output_size[c]).collect();
        let lhs_axes = find_outputs_in_inputs_unique(&output_indices, &lhs_indices);
        let rhs_axes = find_outputs_in_inputs_unique(&output_indices, &rhs_indices);

        TensordotGeneral::from_shapes_and_axis_numbers(&lhs_shape, &rhs_shape, &lhs_axes, &rhs_axes)
    }

    pub fn from_shapes_and_axis_numbers(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Self {
        let num_contracted_axes = lhs_axes.len();
        assert!(num_contracted_axes == rhs_axes.len());
        let lhs_uniques: HashSet<_> = lhs_axes.iter().cloned().collect();
        let rhs_uniques: HashSet<_> = rhs_axes.iter().cloned().collect();
        assert!(num_contracted_axes == lhs_uniques.len());
        assert!(num_contracted_axes == rhs_uniques.len());
        let mut adjusted_lhs_shape = Vec::new();
        let mut adjusted_rhs_shape = Vec::new();

        // Rolls the axes specified in lhs and rhs to the back and front respectively,
        // then calls tensordot_fixed_order(rolled_lhs, rolled_rhs, lhs_axes.len())
        let mut permutation_lhs = Vec::new();
        for (i, &axis_length) in lhs_shape.iter().enumerate() {
            if !(lhs_uniques.contains(&i)) {
                permutation_lhs.push(i);
                adjusted_lhs_shape.push(axis_length);
            }
        }
        for &axis in lhs_axes.iter() {
            permutation_lhs.push(axis);
            adjusted_lhs_shape.push(lhs_shape[axis]);
        }

        // Note: These two for loops are (intentionally!) in the reverse order
        // as they are for LHS.
        let mut permutation_rhs = Vec::new();
        for &axis in rhs_axes.iter() {
            permutation_rhs.push(axis);
            adjusted_rhs_shape.push(rhs_shape[axis]);
        }
        for (i, &axis_length) in rhs_shape.iter().enumerate() {
            if !(rhs_uniques.contains(&i)) {
                permutation_rhs.push(i);
                adjusted_rhs_shape.push(axis_length);
            }
        }

        let lhs_permutation = Permutation::from_indices(&permutation_lhs);
        let rhs_permutation = Permutation::from_indices(&permutation_rhs);
        let tensordot_fixed_position =
            TensordotFixedPosition::from_shapes_and_number_of_contracted_axes(
                &adjusted_lhs_shape,
                &adjusted_rhs_shape,
                num_contracted_axes,
            );

        TensordotGeneral {
            lhs_permutation,
            rhs_permutation,
            tensordot_fixed_position,
        }
    }
}

impl<A> PairContractor<A> for TensordotGeneral {
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
        let permuted_lhs = self.lhs_permutation.view_singleton(lhs);
        let permuted_rhs = self.rhs_permutation.view_singleton(rhs);
        self.tensordot_fixed_position
            .contract_pair(&permuted_lhs, &permuted_rhs)
    }
}

#[derive(Clone, Debug)]
pub struct HadamardProduct {}

impl HadamardProduct {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(lhs_indices, rhs_indices);
        assert_eq!(lhs_indices, output_indices);

        HadamardProduct {}
    }
}

impl<A> PairContractor<A> for HadamardProduct {
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
        lhs * rhs
    }
}

#[derive(Clone, Debug)]
pub struct HadamardProductGeneral {
    lhs_permutation: Permutation,
    rhs_permutation: Permutation,
}

impl HadamardProductGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(lhs_indices.len(), rhs_indices.len());
        assert_eq!(lhs_indices.len(), output_indices.len());

        let lhs_permutation =
            Permutation::from_indices(&find_outputs_in_inputs_unique(output_indices, lhs_indices));
        let rhs_permutation =
            Permutation::from_indices(&find_outputs_in_inputs_unique(output_indices, rhs_indices));

        HadamardProductGeneral {
            lhs_permutation,
            rhs_permutation,
        }
    }
}

impl<A> PairContractor<A> for HadamardProductGeneral {
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
        &self.lhs_permutation.view_singleton(lhs) * &self.rhs_permutation.view_singleton(rhs)
    }
}
