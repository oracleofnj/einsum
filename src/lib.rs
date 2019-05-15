// Copyright 2019 Jared Samet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::{HashMap, HashSet};

use ndarray::prelude::*;
use ndarray::{Data, IxDyn, LinalgScalar};

mod validation;
pub use validation::{
    validate, validate_and_size, validate_and_size_from_shapes, Contraction, OutputSize,
    SizedContraction,
};

mod optimizers;
pub use optimizers::{
    generate_optimized_path, EinsumPath, FirstStep, IntermediateStep, OperandNumPair,
    OptimizationMethod,
};

#[derive(Clone, Debug)]
pub struct UntouchedIndex {
    position: usize,
    output_position: usize,
}

#[derive(Clone, Debug)]
pub struct DiagonalizedIndex {
    positions: Vec<usize>,
    output_position: usize,
}

#[derive(Clone, Debug)]
pub struct SummedIndex {
    positions: Vec<usize>,
}

// TODO: Replace the chars here with usizes and the HashMaps with vecs
#[derive(Clone, Debug)]
pub enum SingletonIndexInfo {
    UntouchedInfo(UntouchedIndex),
    DiagonalizedInfo(DiagonalizedIndex),
    SummedInfo(SummedIndex),
}

#[derive(Clone, Debug)]
pub struct IndexWithSingletonInfo {
    index: char,
    index_info: SingletonIndexInfo,
}

type UntouchedIndexMap = HashMap<char, UntouchedIndex>;
type DiagonalizedIndexMap = HashMap<char, DiagonalizedIndex>;
type SummedIndexMap = HashMap<char, SummedIndex>;

#[derive(Clone, Debug)]
pub struct ClassifiedSingletonContraction {
    input_indices: Vec<IndexWithSingletonInfo>,
    output_indices: Vec<IndexWithSingletonInfo>,
    untouched_indices: UntouchedIndexMap,
    diagonalized_indices: DiagonalizedIndexMap,
    summed_indices: SummedIndexMap,
}

#[derive(Clone, Debug)]
pub struct StackIndex {
    // Which dimension of the LHS tensor does this index correspond to
    lhs_position: usize,
    // Which dimension of the RHS tensor does this index correspond to
    rhs_position: usize,
    // Which dimension in the output does this index get assigned to
    output_position: usize,
}

#[derive(Clone, Debug)]
pub struct ContractedIndex {
    // Which dimension of the LHS tensor does this index correspond to
    lhs_position: usize,
    // Which dimension of the RHS tensor does this index correspond to
    rhs_position: usize,
}

#[derive(Clone, Debug)]
pub enum OuterIndexPosition {
    LHS(usize),
    RHS(usize),
}

#[derive(Clone, Debug)]
pub struct OuterIndex {
    // Which dimension of the input tensor does this index correspond to
    input_position: OuterIndexPosition,
    // Which dimension of the output tensor does this index get assigned to
    output_position: usize,
}

// TODO: Replace the chars here with usizes and the HashMaps with vecs
#[derive(Debug)]
pub enum PairIndexInfo {
    StackInfo(StackIndex),
    ContractedInfo(ContractedIndex),
    OuterInfo(OuterIndex),
}

#[derive(Debug)]
pub struct IndexWithPairInfo {
    index: char,
    index_info: PairIndexInfo,
}

type StackIndexMap = HashMap<char, StackIndex>;
type ContractedIndexMap = HashMap<char, ContractedIndex>;
type OuterIndexMap = HashMap<char, OuterIndex>;

#[derive(Debug)]
pub struct ClassifiedDedupedPairContraction {
    lhs_indices: Vec<IndexWithPairInfo>,
    rhs_indices: Vec<IndexWithPairInfo>,
    output_indices: Vec<IndexWithPairInfo>,
    stack_indices: StackIndexMap,
    contracted_indices: ContractedIndexMap,
    outer_indices: OuterIndexMap,
}

// How this could work:
//
// Make an AST
//
// A "complex" singleton contraction is a (possibly empty) list
// of "simple" singleton contractions
//
// A "simple" singleton contraction is either a diagonalization, a sum,
// or a permutation
//
// A "complex" pair contraction is a singleton contraction for
// the LHS, a singleton contraction for the RHS, and a
// simplified pair contraction
//
// A simplified pair contraction is the a stacked deduped etc.

pub trait ArrayLike<A> {
    fn into_dyn_view(&self) -> ArrayView<A, IxDyn>;
}

impl<A, S, D> ArrayLike<A> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn into_dyn_view(&self) -> ArrayView<A, IxDyn> {
        self.view().into_dyn()
    }
}

// struct MultiAxisIterator<'a, A> {
//     carrying: bool,
//     ndim: usize,
//     // axes: Vec<usize>,
//     renumbered_axes: Vec<usize>,
//     shape: Vec<usize>,
//     positions: Vec<usize>,
//     underlying: &'a ArrayViewD<'a, A>,
//     // subviews: Vec<ArrayViewD<'a, A>>,
// }
//
// impl<'a, A> MultiAxisIterator<'a, A> {
//     fn new(base: &'a ArrayViewD<'a, A>, axes: &[usize]) -> MultiAxisIterator<'a, A> {
//         let ndim = axes.len();
//         // let axes: Vec<usize> = axes.to_vec();
//         let renumbered_axes: Vec<usize> = axes
//             .iter()
//             .enumerate()
//             .map(|(i, &v)| v - axes[0..i].iter().filter(|&&x| x < v).count())
//             .collect();
//         let shape: Vec<usize> = axes
//             .iter()
//             .map(|&x| base.shape().get(x).unwrap())
//             .cloned()
//             .collect();
//         let positions = vec![0; shape.len()];
//
//         // let mut subviews = Vec::new();
//         // let mut axis_iters = Vec::new();
//         //
//         // for (ax_num, &ax) in axes.iter().enumerate() {
//         //     let mut subview = base.view();
//         //     for i in 0..ax_num {
//         //         subview = subview.index_axis_move(Axis(0), 0);
//         //     }
//         //     subviews.push(subview);
//         // }
//
//         MultiAxisIterator {
//             underlying: base,
//             carrying: false,
//             ndim,
//             // axes,
//             renumbered_axes,
//             shape,
//             positions,
//             // subviews,
//         }
//     }
// }
//
// impl<'a, A> Iterator for MultiAxisIterator<'a, A> {
//     type Item = ArrayViewD<'a, A>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if !self.carrying {
//             let mut view = self.underlying.view();
//             for (&ax, &pos) in self.renumbered_axes.iter().zip(&self.positions) {
//                 view = view.index_axis_move(Axis(ax), pos);
//             }
//             self.carrying = true;
//             for i in 0..self.ndim {
//                 let axis = self.ndim - i - 1;
//                 if self.positions[axis] == self.shape[axis] - 1 {
//                     self.positions[axis] = 0;
//                 } else {
//                     self.positions[axis] += 1;
//                     self.carrying = false;
//                     break;
//                 }
//             }
//             Some(view)
//         } else {
//             None
//         }
//     }
// }
//
fn generate_info_vector_for_singleton(
    operand_indices: &[char],
    untouched_indices: &UntouchedIndexMap,
    diagonalized_indices: &DiagonalizedIndexMap,
    summed_indices: &SummedIndexMap,
) -> Vec<IndexWithSingletonInfo> {
    let mut indices = Vec::new();
    for &c in operand_indices {
        if let Some(untouched_info) = untouched_indices.get(&c) {
            indices.push(IndexWithSingletonInfo {
                index: c,
                index_info: SingletonIndexInfo::UntouchedInfo(untouched_info.clone()),
            });
        } else if let Some(diagonalized_info) = diagonalized_indices.get(&c) {
            indices.push(IndexWithSingletonInfo {
                index: c,
                index_info: SingletonIndexInfo::DiagonalizedInfo(diagonalized_info.clone()),
            });
        } else if let Some(summed_info) = summed_indices.get(&c) {
            indices.push(IndexWithSingletonInfo {
                index: c,
                index_info: SingletonIndexInfo::SummedInfo(summed_info.clone()),
            });
        } else {
            panic!();
        }
    }
    indices
}

fn generate_classified_singleton_contraction(
    sized_contraction: &SizedContraction,
) -> ClassifiedSingletonContraction {
    let Contraction {
        ref operand_indices,
        ref output_indices,
        ref summation_indices,
    } = sized_contraction.contraction;

    assert_eq!(operand_indices.len(), 1);

    let mut untouched_indices = HashMap::new();
    let mut diagonalized_indices = HashMap::new();
    let mut summed_indices = HashMap::new();
    let input_chars = &operand_indices[0];

    for (output_position, &c) in output_indices.iter().enumerate() {
        let mut input_positions = Vec::new();
        for (pos, _) in input_chars.iter().enumerate().filter(|&(_, &x)| x == c) {
            input_positions.push(pos);
        }
        match input_positions.len() {
            0 => panic!(),
            1 => {
                untouched_indices.insert(
                    c,
                    UntouchedIndex {
                        position: input_positions[0],
                        output_position,
                    },
                );
            }
            _ => {
                diagonalized_indices.insert(
                    c,
                    DiagonalizedIndex {
                        positions: input_positions,
                        output_position,
                    },
                );
            }
        }
    }

    for &c in summation_indices.iter() {
        let mut input_positions = Vec::new();
        for (pos, _) in input_chars.iter().enumerate().filter(|&(_, &x)| x == c) {
            input_positions.push(pos);
        }
        summed_indices.insert(
            c,
            SummedIndex {
                positions: input_positions,
            },
        );
    }

    let input_indices = generate_info_vector_for_singleton(
        &input_chars,
        &untouched_indices,
        &diagonalized_indices,
        &summed_indices,
    );
    let output_indices = generate_info_vector_for_singleton(
        &output_indices,
        &untouched_indices,
        &diagonalized_indices,
        &summed_indices,
    );

    assert_eq!(
        output_indices.len(),
        untouched_indices.len() + diagonalized_indices.len()
    );
    assert_eq!(summation_indices.len(), summed_indices.len());

    ClassifiedSingletonContraction {
        input_indices,
        output_indices,
        untouched_indices,
        diagonalized_indices,
        summed_indices,
    }
}

fn generate_info_vector_for_pair(
    operand_indices: &[char],
    stack_indices: &StackIndexMap,
    outer_indices: &OuterIndexMap,
    contracted_indices: &ContractedIndexMap,
) -> Vec<IndexWithPairInfo> {
    let mut indices = Vec::new();
    for &c in operand_indices {
        if let Some(stack_info) = stack_indices.get(&c) {
            indices.push(IndexWithPairInfo {
                index: c,
                index_info: PairIndexInfo::StackInfo(stack_info.clone()),
            });
        } else if let Some(outer_info) = outer_indices.get(&c) {
            indices.push(IndexWithPairInfo {
                index: c,
                index_info: PairIndexInfo::OuterInfo(outer_info.clone()),
            });
        } else if let Some(contracted_info) = contracted_indices.get(&c) {
            indices.push(IndexWithPairInfo {
                index: c,
                index_info: PairIndexInfo::ContractedInfo(contracted_info.clone()),
            });
        } else {
            panic!();
        }
    }
    indices
}

fn generate_classified_pair_contraction(
    sized_contraction: &SizedContraction,
) -> ClassifiedDedupedPairContraction {
    let Contraction {
        ref operand_indices,
        ref output_indices,
        ref summation_indices,
    } = sized_contraction.contraction;

    assert!(operand_indices.len() == 2);

    let mut stack_indices = HashMap::new();
    let mut contracted_indices = HashMap::new();
    let mut outer_indices = HashMap::new();
    let lhs_chars = &operand_indices[0];
    let rhs_chars = &operand_indices[1];

    for (output_position, &c) in output_indices.iter().enumerate() {
        match (
            lhs_chars.iter().position(|&x| x == c),
            rhs_chars.iter().position(|&x| x == c),
        ) {
            (Some(lhs_position), Some(rhs_position)) => {
                stack_indices.insert(
                    c,
                    StackIndex {
                        lhs_position,
                        rhs_position,
                        output_position,
                    },
                );
            }
            (Some(lhs_index), None) => {
                outer_indices.insert(
                    c,
                    OuterIndex {
                        input_position: OuterIndexPosition::LHS(lhs_index),
                        output_position,
                    },
                );
            }
            (None, Some(rhs_index)) => {
                outer_indices.insert(
                    c,
                    OuterIndex {
                        input_position: OuterIndexPosition::RHS(rhs_index),
                        output_position,
                    },
                );
            }
            (None, None) => panic!(),
        }
    }

    for &c in summation_indices.iter() {
        match (
            lhs_chars.iter().position(|&x| x == c),
            rhs_chars.iter().position(|&x| x == c),
        ) {
            (Some(lhs_position), Some(rhs_position)) => {
                contracted_indices.insert(
                    c,
                    ContractedIndex {
                        lhs_position,
                        rhs_position,
                    },
                );
            }
            (_, _) => panic!(),
        }
    }

    let lhs_indices = generate_info_vector_for_pair(
        &lhs_chars,
        &stack_indices,
        &outer_indices,
        &contracted_indices,
    );
    let rhs_indices = generate_info_vector_for_pair(
        &rhs_chars,
        &stack_indices,
        &outer_indices,
        &contracted_indices,
    );
    let output_indices = generate_info_vector_for_pair(
        &output_indices,
        &stack_indices,
        &outer_indices,
        &contracted_indices,
    );

    assert_eq!(
        output_indices.len(),
        stack_indices.len() + outer_indices.len()
    );
    assert_eq!(summation_indices.len(), contracted_indices.len());

    ClassifiedDedupedPairContraction {
        lhs_indices,
        rhs_indices,
        output_indices,
        stack_indices,
        contracted_indices,
        outer_indices,
    }
}

pub fn move_output_indices_to_front<'a, A: LinalgScalar>(
    input_indices: &[IndexWithSingletonInfo],
    output_index_order: &[char],
    tensor: &'a ArrayViewD<'a, A>,
) -> ArrayViewD<'a, A> {
    let mut permutation: Vec<usize> = Vec::new();

    for &c in output_index_order {
        let input_pos = input_indices.iter().position(|idx| idx.index == c).unwrap();
        permutation.push(input_pos);
    }

    for (i, idx) in input_indices.iter().enumerate() {
        if let SingletonIndexInfo::SummedInfo(_) = idx.index_info {
            permutation.push(i);
        }
    }

    tensor.view().permuted_axes(permutation)
}

fn einsum_singleton_norepeats<'a, A: LinalgScalar>(
    csc: &ClassifiedSingletonContraction,
    tensor: &'a ArrayViewD<'a, A>,
) -> ArrayD<A> {
    // Handles the case where it's ijk->ik; just sums
    assert_eq!(csc.diagonalized_indices.len(), 0);
    assert_eq!(
        csc.summed_indices
            .values()
            .filter(|x| x.positions.len() != 1)
            .count(),
        0
    );

    let output_index_order: Vec<char> = csc.output_indices.iter().map(|x| x.index).collect();
    let permuted_input =
        move_output_indices_to_front(&csc.input_indices, &output_index_order, tensor);
    if csc.summed_indices.len() == 0 {
        permuted_input.into_owned()
    } else {
        let mut result = permuted_input.sum_axis(Axis(csc.output_indices.len()));
        for _ in 1..csc.summed_indices.len() {
            result = result.sum_axis(Axis(csc.output_indices.len()));
        }
        result
    }
}

// TODO: Replace this by calculating the right dimensions and strides to use
// TODO: Take a &mut ClassifiedSingletonContraction and mutate it
fn diagonalize_singleton<A: LinalgScalar>(
    tensor: &ArrayViewD<A>,
    axes: &[usize],
    destination_axis: usize,
) -> ArrayD<A> {
    // TODO: Replace this now that I understand how to use assign()
    assert!(axes.len() > 0);
    let axis_length = tensor.shape()[axes[0]];
    let slices: Vec<_> = (0..axis_length)
        .map(|i| {
            let mut subview = tensor.view().into_owned();
            let mut foo = Vec::from(axes);
            foo.sort();
            for &j in foo.iter().rev() {
                subview = subview.index_axis_move(Axis(j), i);
            }
            subview.insert_axis(Axis(destination_axis))
        })
        .collect();
    let slice_views: Vec<_> = slices.iter().map(|s| s.view()).collect();
    ndarray::stack(Axis(destination_axis), &slice_views).unwrap()
}

// TODO: Take a &mut ClassifiedSingletonContraction and mutate it instead
// of mutating operand_indices
fn diagonalize_singleton_char<A>(
    tensor: &mut ArrayD<A>,
    operand_indices: &mut Vec<char>,
    repeated_index: char,
) where
    A: LinalgScalar,
{
    let mut new_indices = Vec::new();
    let mut axes = Vec::new();
    new_indices.push(repeated_index);
    for (i, &c) in operand_indices.iter().enumerate() {
        if c != repeated_index {
            new_indices.push(c);
        } else {
            axes.push(i);
        }
    }
    let new_tensor = diagonalize_singleton(&tensor.view(), &axes, 0);

    *tensor = new_tensor;
    *operand_indices = new_indices;
}

// TODO: Figure out correct magic with strides and dimensions
// TODO: Take a ClassifiedSingletonContraction instead of a
//       SizedContraction
pub fn einsum_singleton<'a, A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    tensor: &'a ArrayViewD<'a, A>,
) -> ArrayD<A> {
    // Handles the case where it's iijk->ik; just diagonalization + sums
    assert!(sized_contraction.contraction.operand_indices.len() == 1);
    let mut distinct_elements = HashSet::new();
    let mut repeated_elements = HashSet::new();
    for &c in sized_contraction.contraction.operand_indices[0].iter() {
        if distinct_elements.contains(&c) {
            repeated_elements.insert(c);
        } else {
            distinct_elements.insert(c);
        }
    }

    let no_repeated_elements = repeated_elements.len() == 0;

    if no_repeated_elements {
        let csc = generate_classified_singleton_contraction(sized_contraction);
        einsum_singleton_norepeats(&csc, tensor)
    } else {
        let mut operand_indices = sized_contraction.contraction.operand_indices[0].clone();
        let mut modified_tensor = tensor.view().into_owned();

        for &c in repeated_elements.iter() {
            diagonalize_singleton_char(&mut modified_tensor, &mut operand_indices, c);
        }

        // TODO: Just make the new contraction directly
        let operand_indices_str: String = operand_indices.iter().collect();
        let output_str: String = sized_contraction
            .contraction
            .output_indices
            .iter()
            .collect();
        let new_einsum_string = format!("{}->{}", operand_indices_str, output_str);
        let new_contraction = validate_and_size(&new_einsum_string, &[&modified_tensor]).unwrap();

        let csc = generate_classified_singleton_contraction(&new_contraction);
        einsum_singleton_norepeats(&csc, &modified_tensor.view())
    }
}

fn tensordot_fixed_order<A: LinalgScalar>(
    lhs: &ArrayViewD<A>,
    rhs: &ArrayViewD<A>,
    last_n: usize,
) -> ArrayD<A> {
    // Returns an n-dimensional array where n = |D| + |E| - 2 * last_n.
    // The shape will be (...lhs.shape(:-last_n), ...rhs.shape(last_n:))
    // i.e. if lhs.shape = (3,4,5), rhs.shape = (4,5,6), and last_n=2,
    // the returned array will have shape (3,6).
    //
    // Basically you reshape each one into a 2-D matrix (no matter what
    // the starting size was) and then do a matrix multiplication
    let mut len_uncontracted_lhs = 1;
    let mut len_uncontracted_rhs = 1;
    let mut len_contracted_lhs = 1;
    let mut len_contracted_rhs = 1;
    let mut output_shape = Vec::<usize>::new();
    let num_axes_lhs = lhs.ndim();
    for (axis, &axis_length) in lhs.shape().iter().enumerate() {
        if axis < (num_axes_lhs - last_n) {
            len_uncontracted_lhs *= axis_length;
            output_shape.push(axis_length);
        } else {
            len_contracted_lhs *= axis_length;
        }
    }
    for (axis, &axis_length) in rhs.shape().iter().enumerate() {
        if axis < last_n {
            len_contracted_rhs *= axis_length;
        } else {
            len_uncontracted_rhs *= axis_length;
            output_shape.push(axis_length);
        }
    }
    let matrix1 = Array::from_shape_vec(
        [len_uncontracted_lhs, len_contracted_lhs],
        lhs.iter().cloned().collect(),
    )
    .unwrap();
    let matrix2 = Array::from_shape_vec(
        [len_contracted_rhs, len_uncontracted_rhs],
        rhs.iter().cloned().collect(),
    )
    .unwrap();

    matrix1
        .dot(&matrix2)
        .into_shape(IxDyn(&output_shape))
        .unwrap()
}

pub fn tensordot<A, S, S2, D, E>(
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
    lhs_axes: &[Axis],
    rhs_axes: &[Axis],
) -> ArrayD<A>
where
    A: ndarray::LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    let num_axes = lhs_axes.len();
    assert!(num_axes == rhs_axes.len());
    let mut lhs_startpositions: Vec<_> = lhs_axes.iter().map(|x| x.index()).collect();
    let mut rhs_startpositions: Vec<_> = rhs_axes.iter().map(|x| x.index()).collect();

    // Probably a better way to do this...
    let lhs_uniques: HashSet<_> = lhs_startpositions.iter().cloned().collect();
    let rhs_uniques: HashSet<_> = rhs_startpositions.iter().cloned().collect();
    assert!(num_axes == lhs_uniques.len());
    assert!(num_axes == rhs_uniques.len());

    // Rolls the axes specified in lhs and rhs to the back and front respectively,
    // then calls tensordot_fixed_order(rolled_lhs, rolled_rhs, lhs_axes.len())
    let mut permutation_lhs = Vec::new();
    for i in 0..(lhs.ndim()) {
        if !(lhs_uniques.contains(&i)) {
            permutation_lhs.push(i);
        }
    }
    permutation_lhs.append(&mut lhs_startpositions);
    let rolled_lhs = lhs.view().into_dyn().permuted_axes(permutation_lhs);

    let mut permutation_rhs = Vec::new();
    permutation_rhs.append(&mut rhs_startpositions);
    for i in 0..(rhs.ndim()) {
        if !(rhs_uniques.contains(&i)) {
            permutation_rhs.push(i);
        }
    }
    let rolled_rhs = rhs.view().into_dyn().permuted_axes(permutation_rhs);

    tensordot_fixed_order(&rolled_lhs, &rolled_rhs, lhs_axes.len())
}

fn einsum_pair_allused_nostacks_classified_deduped_indices<A: LinalgScalar>(
    classified_pair_contraction: &ClassifiedDedupedPairContraction,
    lhs: &ArrayViewD<A>,
    rhs: &ArrayViewD<A>,
    output: &mut ArrayViewMutD<A>,
) {
    // Allowed: abc,bce->ae
    // Not allowed: abc,acd -> abd [a in lhs, rhs, and output]
    // Not allowed: abc,bcde -> ae [no d in output]
    // Not allowed: abbc,bce -> ae [repeated b in input]
    // In other words: each index of each tensor is unique,
    // and is either in the other tensor or in the output, but not both

    // If an index is in both tensors, it's part of the tensordot op
    // Otherwise, it's in the output and we need to permute into the correct order
    // afterwards

    assert_eq!(classified_pair_contraction.stack_indices.len(), 0);
    match (
        classified_pair_contraction.contracted_indices.len(),
        classified_pair_contraction.lhs_indices.len(),
        classified_pair_contraction.rhs_indices.len(),
    ) {
        (0, 0, 0) => {
            let scalarize = IxDyn(&[]);
            output[&scalarize] = lhs[&scalarize] * rhs[&scalarize];
        }
        (0, 0, _) => {
            let lhs_0d: A = lhs.first().unwrap().clone();
            let mut both_outer_indices = Vec::new();
            for idx in classified_pair_contraction.rhs_indices.iter() {
                if let PairIndexInfo::OuterInfo(OuterIndex {
                    input_position: OuterIndexPosition::RHS(_),
                    output_position: _,
                }) = idx.index_info
                {
                    both_outer_indices.push(idx.index);
                }
            }
            let permutation: Vec<usize> = classified_pair_contraction
                .output_indices
                .iter()
                .map(|c| {
                    both_outer_indices
                        .iter()
                        .position(|&x| x == c.index)
                        .unwrap()
                })
                .collect();

            output.assign(
                &rhs.mapv(|x| x * lhs_0d)
                    .into_dyn()
                    .view()
                    .permuted_axes(permutation),
            );
        }
        (0, _, 0) => {
            let rhs_0d: A = rhs.first().unwrap().clone();
            let mut both_outer_indices = Vec::new();
            for idx in classified_pair_contraction.lhs_indices.iter() {
                if let PairIndexInfo::OuterInfo(OuterIndex {
                    input_position: OuterIndexPosition::LHS(_),
                    output_position: _,
                }) = idx.index_info
                {
                    both_outer_indices.push(idx.index);
                }
            }
            let permutation: Vec<usize> = classified_pair_contraction
                .output_indices
                .iter()
                .map(|c| {
                    both_outer_indices
                        .iter()
                        .position(|&x| x == c.index)
                        .unwrap()
                })
                .collect();

            output.assign(
                &lhs.mapv(|x| x * rhs_0d)
                    .into_dyn()
                    .view()
                    .permuted_axes(permutation),
            );
        }
        _ => {
            let mut lhs_axes = Vec::new();
            let mut rhs_axes = Vec::new();
            for (_, contracted_index) in classified_pair_contraction.contracted_indices.iter() {
                lhs_axes.push(Axis(contracted_index.lhs_position));
                rhs_axes.push(Axis(contracted_index.rhs_position));
            }

            let mut both_outer_indices = Vec::new();
            for idx in classified_pair_contraction.lhs_indices.iter() {
                if let PairIndexInfo::OuterInfo(OuterIndex {
                    input_position: OuterIndexPosition::LHS(_),
                    output_position: _,
                }) = idx.index_info
                {
                    both_outer_indices.push(idx.index);
                }
            }
            for idx in classified_pair_contraction.rhs_indices.iter() {
                if let PairIndexInfo::OuterInfo(OuterIndex {
                    input_position: OuterIndexPosition::RHS(_),
                    output_position: _,
                }) = idx.index_info
                {
                    both_outer_indices.push(idx.index);
                }
            }
            let permutation: Vec<usize> = classified_pair_contraction
                .output_indices
                .iter()
                .map(|c| {
                    both_outer_indices
                        .iter()
                        .position(|&x| x == c.index)
                        .unwrap()
                })
                .collect();

            output.assign(
                &tensordot(lhs, rhs, &lhs_axes, &rhs_axes)
                    .view()
                    .permuted_axes(permutation),
            );
        }
    }
}

fn move_stack_indices_to_front<A: LinalgScalar>(
    input_indices: &[IndexWithPairInfo],
    stack_index_order: &[char],
    tensor: &ArrayViewD<A>,
) -> ArrayD<A> {
    let mut permutation: Vec<usize> = Vec::new();
    let mut output_shape: Vec<usize> = Vec::new();
    output_shape.push(1);

    for &c in stack_index_order {
        permutation.push(input_indices.iter().position(|idx| idx.index == c).unwrap());
    }

    for (i, (idx, &axis_length)) in input_indices.iter().zip(tensor.shape().iter()).enumerate() {
        if let PairIndexInfo::StackInfo(_) = idx.index_info {
            output_shape[0] *= axis_length;
        } else {
            permutation.push(i);
            output_shape.push(axis_length);
        }
    }

    let temp_result = tensor.view().permuted_axes(permutation);

    Array::from_shape_vec(IxDyn(&output_shape), temp_result.iter().cloned().collect()).unwrap()
}

#[inline(never)]
fn get_intermediate_result<A: LinalgScalar>(
    temp_shape: &[usize],
    lhs_reshaped: &ArrayD<A>,
    rhs_reshaped: &ArrayD<A>,
    new_cdpc: &ClassifiedDedupedPairContraction,
) -> ArrayD<A> {
    let mut temp_result: ArrayD<A> = Array::zeros(IxDyn(&temp_shape));
    let mut lhs_iter = lhs_reshaped.outer_iter();
    let mut rhs_iter = rhs_reshaped.outer_iter();
    for mut output_subview in temp_result.outer_iter_mut() {
        let lhs_subview = lhs_iter.next().unwrap();
        let rhs_subview = rhs_iter.next().unwrap();
        // let mut output_subview = temp_result.index_axis_mut(Axis(0), i);
        // let lhs_subview = lhs_reshaped.index_axis(Axis(0), i);
        // let rhs_subview = rhs_reshaped.index_axis(Axis(0), i);
        einsum_pair_allused_nostacks_classified_deduped_indices(
            &new_cdpc,
            &lhs_subview,
            &rhs_subview,
            &mut output_subview,
        );
    }
    temp_result
}

fn einsum_pair_allused_deduped_indices<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    lhs: &ArrayViewD<A>,
    rhs: &ArrayViewD<A>,
) -> ArrayD<A> {
    // Allowed: abc,bce->ae
    // Allowed: abc,acd -> abd [a in lhs, rhs, and output]
    // Not allowed: abc,bcde -> ae [no d in output]
    // Not allowed: abbc,bce -> ae [repeated b in input]
    // In other words: each index of each tensor is unique,
    // and is either in the other tensor or in the output

    let cpc = generate_classified_pair_contraction(sized_contraction);

    if cpc.stack_indices.len() == 0 {
        let result_shape: Vec<usize> = sized_contraction
            .contraction
            .output_indices
            .iter()
            .map(|c| sized_contraction.output_size[c])
            .collect();
        let mut result: ArrayD<A> = Array::zeros(IxDyn(&result_shape));
        einsum_pair_allused_nostacks_classified_deduped_indices(
            &cpc,
            lhs,
            rhs,
            &mut result.view_mut(),
        );
        result
    } else {
        // What do we have to do?
        // (1) Permute the stack indices to the front of LHS and RHS and
        //     Reshape into (N, ...non-stacked LHS shape), (N, ...non-stacked RHS shape)
        let mut stack_index_order = Vec::new();
        for idx in cpc.output_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
                stack_index_order.push(idx.index);
            }
        }
        // OLD:
        let lhs_reshaped = move_stack_indices_to_front(&cpc.lhs_indices, &stack_index_order, &lhs);
        let rhs_reshaped = move_stack_indices_to_front(&cpc.rhs_indices, &stack_index_order, &rhs);

        // NEW:
        let mut lhs_stack_axes = Vec::new();
        let mut rhs_stack_axes = Vec::new();
        for &c in stack_index_order.iter() {
            lhs_stack_axes.push(
                cpc.lhs_indices
                    .iter()
                    .position(|idx| idx.index == c)
                    .unwrap(),
            );
            rhs_stack_axes.push(
                cpc.rhs_indices
                    .iter()
                    .position(|idx| idx.index == c)
                    .unwrap(),
            );
        }
        // let lhs_dyn_view = lhs.view().into_dyn();
        // let rhs_dyn_view = rhs.view().into_dyn();
        // let mut lhs_iter = MultiAxisIterator::new(&lhs_dyn_view, &lhs_stack_axes);
        // let mut rhs_iter = MultiAxisIterator::new(&rhs_dyn_view, &rhs_stack_axes);

        // (2) Construct the non-stacked ClassifiedDedupedPairContraction
        let mut unstacked_lhs_chars = Vec::new();
        let mut unstacked_rhs_chars = Vec::new();
        let mut unstacked_output_chars = Vec::new();
        let mut summation_chars = Vec::new();
        let mut num_subviews = 1;
        for idx in cpc.lhs_indices.iter() {
            match idx.index_info {
                PairIndexInfo::OuterInfo(_) => {
                    unstacked_lhs_chars.push(idx.index);
                }
                PairIndexInfo::ContractedInfo(_) => {
                    unstacked_lhs_chars.push(idx.index);
                    summation_chars.push(idx.index);
                }
                PairIndexInfo::StackInfo(_) => {
                    num_subviews *= sized_contraction.output_size[&idx.index];
                }
            }
        }
        for idx in cpc.rhs_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                unstacked_rhs_chars.push(idx.index);
            }
        }
        for idx in cpc.output_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                unstacked_output_chars.push(idx.index);
            }
        }
        let new_sized_contraction = SizedContraction {
            contraction: Contraction {
                operand_indices: vec![unstacked_lhs_chars.clone(), unstacked_rhs_chars.clone()],
                output_indices: unstacked_output_chars,
                summation_indices: summation_chars,
            },
            output_size: HashMap::new(),
        };
        let new_cdpc = generate_classified_pair_contraction(&new_sized_contraction);

        // (3) for i = 0..N, assign the result of einsum_pair_allused_nostacks_classified_deduped_indices
        //     to unshaped_result[i]
        let mut temp_shape: Vec<usize> = Vec::new();
        temp_shape.push(num_subviews);
        let mut final_shape: Vec<usize> = Vec::new();
        let mut intermediate_indices: Vec<char> = Vec::new();
        for (idx, c) in cpc
            .output_indices
            .iter()
            .zip(sized_contraction.contraction.output_indices.iter())
        {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
                final_shape.push(sized_contraction.output_size[c]);
                intermediate_indices.push(*c);
            }
        }
        for (idx, c) in cpc
            .output_indices
            .iter()
            .zip(sized_contraction.contraction.output_indices.iter())
        {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                temp_shape.push(sized_contraction.output_size[c]);
                final_shape.push(sized_contraction.output_size[c]);
                intermediate_indices.push(*c);
            }
        }
        let temp_result =
            get_intermediate_result(&temp_shape, &lhs_reshaped, &rhs_reshaped, &new_cdpc);
        // for (mut output_subview, (lhs_subview, rhs_subview)) in temp_result
        //     .outer_iter_mut()
        //     .zip(lhs_reshaped.outer_iter().zip(rhs_reshaped.outer_iter()))
        // {
        //     // let lhs_subview = lhs_iter.next().unwrap();
        //     // let rhs_subview = rhs_iter.next().unwrap();
        //     let output = einsum_pair_allused_nostacks_classified_deduped_indices(
        //         &new_cdpc,
        //         &lhs_subview,
        //         &rhs_subview,
        //     );
        //     output_subview.assign(&output);
        // }
        //
        // (6) Permute into correct order
        let mut permutation: Vec<usize> = Vec::new();
        for &c in intermediate_indices.iter() {
            permutation.push(
                sized_contraction
                    .contraction
                    .output_indices
                    .iter()
                    .position(|&x| x == c)
                    .unwrap(),
            );
        }

        temp_result
            .into_shape(IxDyn(&final_shape))
            .unwrap()
            .permuted_axes(permutation)
    }

}

fn einsum_pair<'a, A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    lhs: &'a ArrayViewD<'a, A>,
    rhs: &'a ArrayViewD<'a, A>,
) -> ArrayD<A> {
    // If we have abc,bcde -> ae [no d in output] or abbc,bce -> ae [repeated b in input],
    // collapse the offending tensor before delegating to einsum_pair_allused_deduped_indices
    // In other words: collapse each tensor so that each index of each tensor is unique,
    // and is either in the other tensor or in the output

    assert_eq!(sized_contraction.contraction.operand_indices.len(), 2);
    let mut lhs_existing: Vec<char> = sized_contraction.contraction.operand_indices[0].clone();
    let mut rhs_existing: Vec<char> = sized_contraction.contraction.operand_indices[1].clone();

    let lhs_uniques: HashSet<char> = lhs_existing.iter().cloned().collect();
    let rhs_uniques: HashSet<char> = rhs_existing.iter().cloned().collect();
    let output_uniques: HashSet<char> = sized_contraction
        .contraction
        .output_indices
        .iter()
        .cloned()
        .collect();

    let rhs_and_output: HashSet<char> = rhs_uniques.union(&output_uniques).cloned().collect();
    let lhs_and_output: HashSet<char> = lhs_uniques.union(&output_uniques).cloned().collect();

    let mut lhs_desired: Vec<char> = lhs_uniques.intersection(&rhs_and_output).cloned().collect();
    let mut rhs_desired: Vec<char> = rhs_uniques.intersection(&lhs_and_output).cloned().collect();

    lhs_desired.sort();
    rhs_desired.sort();
    lhs_existing.sort();
    rhs_existing.sort();

    let simplify_lhs = !(lhs_desired == lhs_existing);
    let simplify_rhs = !(rhs_desired == rhs_existing);

    match (simplify_lhs, simplify_rhs) {
        (false, false) => einsum_pair_allused_deduped_indices(sized_contraction, lhs, rhs),
        (true, true) => {
            let lhs_str: String = sized_contraction.contraction.operand_indices[0]
                .iter()
                .collect();
            let new_lhs_str: String = lhs_desired.iter().collect();
            let einsum_string_lhs = format!("{}->{}", &lhs_str, new_lhs_str);
            let rhs_str: String = sized_contraction.contraction.operand_indices[1]
                .iter()
                .collect();
            let new_rhs_str: String = rhs_desired.iter().collect();
            let einsum_string_rhs = format!("{}->{}", &rhs_str, new_rhs_str);
            let output_str: String = sized_contraction
                .contraction
                .output_indices
                .iter()
                .collect();
            let new_einsum_string = format!("{},{}->{}", new_lhs_str, new_rhs_str, output_str);
            let sc = validate_and_size(&einsum_string_lhs, &[lhs]).unwrap();
            let lhs_collapsed = einsum_singleton(&sc, lhs);
            let sc = validate_and_size(&einsum_string_rhs, &[rhs]).unwrap();
            let rhs_collapsed = einsum_singleton(&sc, rhs);
            let sc =
                validate_and_size(&new_einsum_string, &[&lhs_collapsed, &rhs_collapsed]).unwrap();
            einsum_pair_allused_deduped_indices(&sc, &lhs_collapsed.view(), &rhs_collapsed.view())
        }
        (true, false) => {
            let lhs_str: String = sized_contraction.contraction.operand_indices[0]
                .iter()
                .collect();
            let new_lhs_str: String = lhs_desired.iter().collect();
            let einsum_string_lhs = format!("{}->{}", &lhs_str, new_lhs_str);
            let new_rhs_str: String = sized_contraction.contraction.operand_indices[1]
                .iter()
                .collect();
            let output_str: String = sized_contraction
                .contraction
                .output_indices
                .iter()
                .collect();
            let new_einsum_string = format!("{},{}->{}", new_lhs_str, new_rhs_str, output_str);
            let sc = validate_and_size(&einsum_string_lhs, &[lhs]).unwrap();
            let lhs_collapsed = einsum_singleton(&sc, lhs);
            let sc = validate_and_size(&new_einsum_string, &[&lhs_collapsed, rhs]).unwrap();
            einsum_pair_allused_deduped_indices(&sc, &lhs_collapsed.view(), rhs)
        }
        (false, true) => {
            let new_lhs_str: String = sized_contraction.contraction.operand_indices[0]
                .iter()
                .collect();
            let rhs_str: String = sized_contraction.contraction.operand_indices[1]
                .iter()
                .collect();
            let new_rhs_str: String = rhs_desired.iter().collect();
            let einsum_string_rhs = format!("{}->{}", rhs_str, new_rhs_str);
            let output_str: String = sized_contraction
                .contraction
                .output_indices
                .iter()
                .collect();
            let new_einsum_string = format!("{},{}->{}", new_lhs_str, new_rhs_str, output_str);
            let sc = validate_and_size(&einsum_string_rhs, &[rhs]).unwrap();
            let rhs_collapsed = einsum_singleton(&sc, rhs);
            let sc = validate_and_size(&new_einsum_string, &[lhs, &rhs_collapsed]).unwrap();
            einsum_pair_allused_deduped_indices(&sc, lhs, &rhs_collapsed.view())
        }
    }
}

pub fn einsum_path<A>(path: &EinsumPath, operands: &[&ArrayLike<A>]) -> ArrayD<A>
where
    A: LinalgScalar,
{
    let EinsumPath {
        first_step:
            FirstStep {
                ref sized_contraction,
                ref operand_nums,
            },
        ref remaining_steps,
    } = path;

    let mut result = match operand_nums {
        None => einsum_singleton(sized_contraction, &(operands[0].into_dyn_view())),
        Some(OperandNumPair { lhs, rhs }) => einsum_pair(
            sized_contraction,
            &(operands[*lhs].into_dyn_view()),
            &(operands[*rhs].into_dyn_view()),
        ),
    };
    for step in remaining_steps.iter() {
        let IntermediateStep {
            ref sized_contraction,
            ref rhs_num,
        } = step;
        result = einsum_pair(
            &sized_contraction,
            &result.view(),
            &(operands[*rhs_num].into_dyn_view()),
        );
    }
    result
}

pub fn einsum_sc<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    operands: &[&ArrayLike<A>],
) -> ArrayD<A> {
    let path = generate_optimized_path(sized_contraction, OptimizationMethod::Naive);
    einsum_path(&path, operands)
}

pub fn einsum<A: LinalgScalar>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<ArrayD<A>, &'static str> {
    let sized_contraction = validate_and_size(input_string, operands)?;
    Ok(einsum_sc(&sized_contraction, operands))
}

mod wasm_bindings;
pub use wasm_bindings::{
    slow_einsum_with_flattened_operands_as_json_string_as_json,
    validate_and_size_from_shapes_as_string_as_json, validate_as_json,
};

mod slow_versions;
pub use slow_versions::{slow_einsum, slow_einsum_given_sized_contraction};
