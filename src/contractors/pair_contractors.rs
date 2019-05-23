use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::HashSet;

use super::{PairContractor, Permutation, SingletonContractor, SingletonViewer};
use crate::SizedContraction;

fn maybe_find_outputs_in_inputs_unique(
    output_indices: &[char],
    input_indices: &[char],
) -> Vec<Option<usize>> {
    output_indices
        .iter()
        .map(|&output_char| {
            let input_pos = input_indices
                .iter()
                .position(|&input_char| input_char == output_char);
            if input_pos.is_some() {
                assert!(input_indices
                    .iter()
                    .skip(input_pos.unwrap() + 1)
                    .position(|&input_char| input_char == output_char)
                    .is_none());
            }
            input_pos
        })
        .collect()
}

fn find_outputs_in_inputs_unique(output_indices: &[char], input_indices: &[char]) -> Vec<usize> {
    maybe_find_outputs_in_inputs_unique(output_indices, input_indices)
        .iter()
        .map(|x| x.unwrap())
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
    fn new(sc: &SizedContraction) -> Self {
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

// TODO: Micro-optimization possible: Have a version without the final permutation,
// which clones the array
#[derive(Clone, Debug)]
pub struct TensordotGeneral {
    lhs_permutation: Permutation,
    rhs_permutation: Permutation,
    tensordot_fixed_position: TensordotFixedPosition,
    output_permutation: Permutation,
}

impl TensordotGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let contracted_indices = &sc.contraction.summation_indices;
        let output_indices = &sc.contraction.output_indices;
        let lhs_shape: Vec<usize> = lhs_indices.iter().map(|c| sc.output_size[c]).collect();
        let rhs_shape: Vec<usize> = rhs_indices.iter().map(|c| sc.output_size[c]).collect();

        TensordotGeneral::from_shapes_and_indices(
            &lhs_shape,
            &rhs_shape,
            lhs_indices,
            &rhs_indices,
            &contracted_indices,
            &output_indices,
        )
    }

    fn from_shapes_and_indices(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        lhs_indices: &[char],
        rhs_indices: &[char],
        contracted_indices: &[char],
        output_indices: &[char],
    ) -> Self {
        let lhs_contracted_axes = find_outputs_in_inputs_unique(&contracted_indices, &lhs_indices);
        let rhs_contracted_axes = find_outputs_in_inputs_unique(&contracted_indices, &rhs_indices);
        let mut uncontracted_chars: Vec<char> = lhs_indices
            .iter()
            .filter(|&&input_char| {
                output_indices
                    .iter()
                    .position(|&output_char| input_char == output_char)
                    .is_some()
            })
            .cloned()
            .collect();
        let mut rhs_uncontracted_chars: Vec<char> = rhs_indices
            .iter()
            .filter(|&&input_char| {
                output_indices
                    .iter()
                    .position(|&output_char| input_char == output_char)
                    .is_some()
            })
            .cloned()
            .collect();
        uncontracted_chars.append(&mut rhs_uncontracted_chars);
        let output_order = find_outputs_in_inputs_unique(&output_indices, &uncontracted_chars);

        TensordotGeneral::from_shapes_and_axis_numbers(
            &lhs_shape,
            &rhs_shape,
            &lhs_contracted_axes,
            &rhs_contracted_axes,
            &output_order,
        )
    }

    pub fn from_shapes_and_axis_numbers(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: &[usize],
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

        let output_permutation = Permutation::from_indices(&output_order);

        TensordotGeneral {
            lhs_permutation,
            rhs_permutation,
            tensordot_fixed_position,
            output_permutation,
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
        let tensordotted = self
            .tensordot_fixed_position
            .contract_pair(&permuted_lhs, &permuted_rhs);
        self.output_permutation
            .contract_singleton(&tensordotted.view())
    }
}

#[derive(Clone, Debug)]
pub struct HadamardProduct {}

impl HadamardProduct {
    fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(lhs_indices, rhs_indices);
        assert_eq!(lhs_indices, output_indices);

        HadamardProduct {}
    }

    fn from_nothing() -> Self {
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
    hadamard_product: HadamardProduct,
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
        let hadamard_product = HadamardProduct::from_nothing();

        HadamardProductGeneral {
            lhs_permutation,
            rhs_permutation,
            hadamard_product,
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
        self.hadamard_product.contract_pair(
            &self.lhs_permutation.view_singleton(lhs),
            &self.rhs_permutation.view_singleton(rhs),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ScalarMatrixProduct {}

impl ScalarMatrixProduct {
    fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(lhs_indices.len(), 0);
        assert_eq!(output_indices, rhs_indices);

        ScalarMatrixProduct {}
    }

    pub fn from_nothing() -> Self {
        ScalarMatrixProduct {}
    }
}

impl<A> PairContractor<A> for ScalarMatrixProduct {
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
        let lhs_0d: A = lhs.first().unwrap().clone();
        rhs.mapv(|x| x * lhs_0d)
    }
}

#[derive(Clone, Debug)]
pub struct ScalarMatrixProductGeneral {
    rhs_permutation: Permutation,
    scalar_matrix_product: ScalarMatrixProduct,
}

impl ScalarMatrixProductGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(lhs_indices.len(), 0);
        assert_eq!(rhs_indices.len(), output_indices.len());

        ScalarMatrixProductGeneral::from_indices(rhs_indices, output_indices)
    }

    pub fn from_indices(input_indices: &[char], output_indices: &[char]) -> Self {
        let rhs_permutation = Permutation::from_indices(&find_outputs_in_inputs_unique(
            output_indices,
            input_indices,
        ));
        let scalar_matrix_product = ScalarMatrixProduct::from_nothing();

        ScalarMatrixProductGeneral {
            rhs_permutation,
            scalar_matrix_product,
        }
    }
}

impl<A> PairContractor<A> for ScalarMatrixProductGeneral {
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
        self.scalar_matrix_product
            .contract_pair(lhs, &self.rhs_permutation.view_singleton(rhs))
    }
}

#[derive(Clone, Debug)]
pub struct MatrixScalarProduct {}

impl MatrixScalarProduct {
    fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(rhs_indices.len(), 0);
        assert_eq!(output_indices, lhs_indices);

        MatrixScalarProduct {}
    }

    pub fn from_nothing() -> Self {
        MatrixScalarProduct {}
    }
}

impl<A> PairContractor<A> for MatrixScalarProduct {
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
        let rhs_0d: A = rhs.first().unwrap().clone();
        lhs.mapv(|x| x * rhs_0d)
    }
}

#[derive(Clone, Debug)]
pub struct MatrixScalarProductGeneral {
    lhs_permutation: Permutation,
    matrix_scalar_product: MatrixScalarProduct,
}

impl MatrixScalarProductGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;
        assert_eq!(rhs_indices.len(), 0);
        assert_eq!(lhs_indices.len(), output_indices.len());

        MatrixScalarProductGeneral::from_indices(lhs_indices, output_indices)
    }

    pub fn from_indices(input_indices: &[char], output_indices: &[char]) -> Self {
        let lhs_permutation = Permutation::from_indices(&find_outputs_in_inputs_unique(
            output_indices,
            input_indices,
        ));
        let matrix_scalar_product = MatrixScalarProduct::from_nothing();

        MatrixScalarProductGeneral {
            lhs_permutation,
            matrix_scalar_product,
        }
    }
}

impl<A> PairContractor<A> for MatrixScalarProductGeneral {
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
        self.matrix_scalar_product
            .contract_pair(&self.lhs_permutation.view_singleton(lhs), rhs)
    }
}

#[derive(Clone, Debug)]
pub struct BroadcastProductGeneral {
    lhs_permutation: Permutation,
    rhs_permutation: Permutation,
    lhs_insertions: Vec<usize>,
    rhs_insertions: Vec<usize>,
    output_shape: IxDyn,
    hadamard_product: HadamardProduct,
}

impl BroadcastProductGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;

        let maybe_lhs_indices = maybe_find_outputs_in_inputs_unique(&output_indices, &lhs_indices);
        let maybe_rhs_indices = maybe_find_outputs_in_inputs_unique(&output_indices, &rhs_indices);
        let lhs_indices: Vec<usize> = maybe_lhs_indices
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();
        let rhs_indices: Vec<usize> = maybe_rhs_indices
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();
        let lhs_insertions: Vec<usize> = maybe_lhs_indices
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_none())
            .map(|(i, _)| i)
            .collect();
        let rhs_insertions: Vec<usize> = maybe_rhs_indices
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_none())
            .map(|(i, _)| i)
            .collect();
        let lhs_permutation = Permutation::from_indices(&lhs_indices);
        let rhs_permutation = Permutation::from_indices(&rhs_indices);
        let output_sizes: Vec<usize> = output_indices.iter().map(|c| sc.output_size[c]).collect();
        let output_shape = IxDyn(&output_sizes);
        let hadamard_product = HadamardProduct::from_nothing();

        BroadcastProductGeneral {
            lhs_permutation,
            rhs_permutation,
            lhs_insertions,
            rhs_insertions,
            output_shape,
            hadamard_product,
        }
    }
}

impl<A> PairContractor<A> for BroadcastProductGeneral {
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
        let mut adjusted_lhs = self.lhs_permutation.view_singleton(lhs);
        let mut adjusted_rhs = self.rhs_permutation.view_singleton(rhs);
        for &i in self.lhs_insertions.iter() {
            adjusted_lhs = adjusted_lhs.insert_axis(Axis(i));
        }
        for &i in self.rhs_insertions.iter() {
            adjusted_rhs = adjusted_rhs.insert_axis(Axis(i));
        }
        let broadcast_lhs = adjusted_lhs.broadcast(self.output_shape.clone()).unwrap();
        let broadcast_rhs = adjusted_rhs.broadcast(self.output_shape.clone()).unwrap();
        self.hadamard_product
            .contract_pair(&broadcast_lhs, &broadcast_rhs)
    }
}

// TODO: Micro-optimization: Have a version without the output permutation,
// which clones the array
#[derive(Clone, Debug)]
pub struct StackedTensordotGeneral {
    lhs_permutation: Permutation,
    rhs_permutation: Permutation,
    lhs_output_shape: Vec<usize>,
    rhs_output_shape: Vec<usize>,
    intermediate_shape: Vec<usize>,
    tensordot_fixed_position: TensordotFixedPosition,
    output_shape: Vec<usize>,
    output_permutation: Permutation,
}

impl StackedTensordotGeneral {
    pub fn new(sc: &SizedContraction) -> Self {
        let mut lhs_order = Vec::new();
        let mut rhs_order = Vec::new();
        let mut lhs_output_shape = Vec::new();
        let mut rhs_output_shape = Vec::new();
        let mut intermediate_shape = Vec::new();

        assert_eq!(sc.contraction.operand_indices.len(), 2);
        let lhs_indices = &sc.contraction.operand_indices[0];
        let rhs_indices = &sc.contraction.operand_indices[1];
        let output_indices = &sc.contraction.output_indices;

        let maybe_lhs_axes = maybe_find_outputs_in_inputs_unique(&output_indices, &lhs_indices);
        let maybe_rhs_axes = maybe_find_outputs_in_inputs_unique(&output_indices, &rhs_indices);
        let mut lhs_stack_axes = Vec::new();
        let mut rhs_stack_axes = Vec::new();
        let mut stack_indices = Vec::new();
        let mut lhs_outer_axes = Vec::new();
        let mut lhs_outer_indices = Vec::new();
        let mut rhs_outer_axes = Vec::new();
        let mut rhs_outer_indices = Vec::new();
        let mut lhs_contracted_axes = Vec::new();
        let mut rhs_contracted_axes = Vec::new();
        let mut intermediate_indices = Vec::new();

        lhs_output_shape.push(1);
        rhs_output_shape.push(1);

        for ((&maybe_lhs_pos, &maybe_rhs_pos), &output_char) in maybe_lhs_axes
            .iter()
            .zip(maybe_rhs_axes.iter())
            .zip(output_indices.iter())
        {
            match (maybe_lhs_pos, maybe_rhs_pos) {
                (Some(lhs_pos), Some(rhs_pos)) => {
                    lhs_stack_axes.push(lhs_pos);
                    rhs_stack_axes.push(rhs_pos);
                    stack_indices.push(output_char);
                    lhs_output_shape[0] *= sc.output_size[&output_char];
                    rhs_output_shape[0] *= sc.output_size[&output_char];
                }
                (Some(lhs_pos), None) => {
                    lhs_outer_axes.push(lhs_pos);
                    lhs_outer_indices.push(output_char);
                    lhs_output_shape.push(sc.output_size[&output_char]);
                }
                (None, Some(rhs_pos)) => {
                    rhs_outer_axes.push(rhs_pos);
                    rhs_outer_indices.push(output_char);
                    rhs_output_shape.push(sc.output_size[&output_char]);
                }
                (None, None) => {
                    panic!() // Output char must be either in lhs or rhs
                }
            }
        }

        for (lhs_pos, &lhs_char) in lhs_indices.iter().enumerate() {
            if let None = output_indices
                .iter()
                .position(|&output_char| output_char == lhs_char)
            {
                // Contracted index
                lhs_contracted_axes.push(lhs_pos);
                // Must be in RHS if it's not in output
                rhs_contracted_axes.push(
                    rhs_indices
                        .iter()
                        .position(|&rhs_char| rhs_char == lhs_char)
                        .unwrap(),
                );
                lhs_output_shape.push(sc.output_size[&lhs_char]);
                rhs_output_shape.push(sc.output_size[&lhs_char]);
            }
        }
        // What order do we want the axes in?
        //
        // LHS: Stack axes, outer axes, contracted axes
        // RHS: Stack axes, contracted axes, outer axes

        lhs_order.append(&mut lhs_stack_axes.clone());
        lhs_order.append(&mut lhs_outer_axes.clone());
        lhs_order.append(&mut lhs_contracted_axes.clone());
        rhs_order.append(&mut rhs_stack_axes.clone());
        rhs_order.append(&mut rhs_contracted_axes.clone());
        rhs_order.append(&mut rhs_outer_axes.clone());

        // What order will the intermediate output indices be in?
        // Stack indices, lhs outer indices, rhs outer indices
        intermediate_indices.append(&mut stack_indices.clone());
        intermediate_indices.append(&mut lhs_outer_indices.clone());
        intermediate_indices.append(&mut rhs_outer_indices.clone());

        assert_eq!(lhs_output_shape[0], rhs_output_shape[0]);
        intermediate_shape.push(lhs_output_shape[0]);
        for lhs_char in lhs_outer_indices.iter() {
            intermediate_shape.push(sc.output_size[lhs_char]);
        }
        for rhs_char in rhs_outer_indices.iter() {
            intermediate_shape.push(sc.output_size[rhs_char]);
        }

        let output_order = find_outputs_in_inputs_unique(&output_indices, &intermediate_indices);
        let output_shape = intermediate_indices
            .iter()
            .map(|c| sc.output_size[c])
            .collect();

        let tensordot_fixed_position =
            TensordotFixedPosition::from_shapes_and_number_of_contracted_axes(
                &lhs_output_shape[1..],
                &rhs_output_shape[1..],
                lhs_contracted_axes.len(),
            );
        let lhs_permutation = Permutation::from_indices(&lhs_order);
        let rhs_permutation = Permutation::from_indices(&rhs_order);
        let output_permutation = Permutation::from_indices(&output_order);
        StackedTensordotGeneral {
            lhs_permutation,
            rhs_permutation,
            lhs_output_shape,
            rhs_output_shape,
            intermediate_shape,
            tensordot_fixed_position,
            output_shape,
            output_permutation,
        }
    }
}

impl<A> PairContractor<A> for StackedTensordotGeneral {
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
        let lhs_permuted = self.lhs_permutation.view_singleton(lhs);
        let lhs_reshaped = Array::from_shape_vec(
            IxDyn(&self.lhs_output_shape),
            lhs_permuted.iter().cloned().collect(),
        )
        .unwrap();
        let rhs_permuted = self.rhs_permutation.view_singleton(rhs);
        let rhs_reshaped = Array::from_shape_vec(
            IxDyn(&self.rhs_output_shape),
            rhs_permuted.iter().cloned().collect(),
        )
        .unwrap();
        let mut intermediate_result: ArrayD<A> = Array::zeros(IxDyn(&self.intermediate_shape));
        let mut lhs_iter = lhs_reshaped.outer_iter();
        let mut rhs_iter = rhs_reshaped.outer_iter();
        for mut output_subview in intermediate_result.outer_iter_mut() {
            let lhs_subview = lhs_iter.next().unwrap();
            let rhs_subview = rhs_iter.next().unwrap();
            self.tensordot_fixed_position.contract_and_assign_pair(
                &lhs_subview,
                &rhs_subview,
                &mut output_subview,
            );
        }
        let intermediate_reshaped = intermediate_result
            .into_shape(IxDyn(&self.output_shape))
            .unwrap();
        self.output_permutation
            .contract_singleton(&intermediate_reshaped.view())
    }
}
