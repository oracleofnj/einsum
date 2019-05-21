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

use crate::classifiers::ClassifiedDedupedPairContraction;
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

mod contractors;
pub use contractors::{
    BroadcastProductGeneral, HadamardProductGeneral, MatrixScalarProductGeneral, PairContractor,
    ScalarMatrixProductGeneral, SingletonContraction, SingletonContractor, TensordotGeneral,
};

mod classifiers;
use classifiers::*;

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

pub fn einsum_singleton<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    tensor: &ArrayViewD<A>,
) -> ArrayD<A> {
    let csc = SingletonContraction::new(&sized_contraction);
    csc.contract_singleton(tensor)
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
    inner_contractor: &Box<dyn PairContractor<A>>,
) -> ArrayD<A> {
    let mut temp_result: ArrayD<A> = Array::zeros(IxDyn(&temp_shape));
    let mut lhs_iter = lhs_reshaped.outer_iter();
    let mut rhs_iter = rhs_reshaped.outer_iter();
    for mut output_subview in temp_result.outer_iter_mut() {
        let lhs_subview = lhs_iter.next().unwrap();
        let rhs_subview = rhs_iter.next().unwrap();
        inner_contractor.contract_and_assign_pair(&lhs_subview, &rhs_subview, &mut output_subview);
    }
    temp_result
}

fn get_pair_contractor<A>(sized_contraction: &SizedContraction) -> Box<dyn PairContractor<A>> {
    let cpc = generate_classified_pair_contraction(sized_contraction);

    match (
        cpc.outer_indices.len(),
        cpc.contracted_indices.len(),
        cpc.lhs_indices.len(),
        cpc.rhs_indices.len(),
    ) {
        // (0, 0, _, _) => Box::new(HadamardProductGeneral::new(&sized_contraction)),
        // (_, 0, 0, _) => Box::new(ScalarMatrixProductGeneral::new(&sized_contraction)),
        // (_, 0, _, 0) => Box::new(MatrixScalarProductGeneral::new(&sized_contraction)),
        (0, 0, _, _) => panic!(),
        (_, 0, 0, _) => panic!(),
        (_, 0, _, 0) => panic!(),
        (_, _, _, _) => Box::new(TensordotGeneral::new(&sized_contraction)),
    }
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

    match (
        cpc.stack_indices.len(),
        cpc.outer_indices.len(),
        cpc.contracted_indices.len(),
        cpc.lhs_indices.len(),
        cpc.rhs_indices.len(),
        cpc.lhs_indices.len() > cpc.stack_indices.len(),
        cpc.rhs_indices.len() > cpc.stack_indices.len(),
    ) {
        (_, 0, 0, _, _, _, _) => {
            let hadamarder = HadamardProductGeneral::new(&sized_contraction);
            hadamarder.contract_pair(lhs, rhs)
        }
        (0, _, 0, 0, _, _, _) => {
            let scalermatrixer = ScalarMatrixProductGeneral::new(&sized_contraction);
            scalermatrixer.contract_pair(lhs, rhs)
        }
        (0, _, 0, _, 0, _, _) => {
            let matrixscalerer = MatrixScalarProductGeneral::new(&sized_contraction);
            matrixscalerer.contract_pair(lhs, rhs)
        }
        (0, _, _, _, _, _, _) => {
            let tensordotter = TensordotGeneral::new(&sized_contraction);
            tensordotter.contract_pair(lhs, rhs)
        }
        (_, _, 0, _, _, _, _) => {
            let contractor = BroadcastProductGeneral::new(&sized_contraction);
            contractor.contract_pair(lhs, rhs)
        }
        // (_, _, 0, _, _, false, _) => {
        //     let contractor = BroadcastProductGeneral::new(&sized_contraction);
        //     contractor.contract_pair(lhs, rhs)
        // }
        // (_, _, 0, _, _, _, false) => {
        //     let contractor = BroadcastProductGeneral::new(&sized_contraction);
        //     contractor.contract_pair(lhs, rhs)
        // }
        (_, _, _, _, _, _, _) => {
            // (1) Permute the stack indices to the front of LHS and RHS and
            //     Reshape into (N, ...non-stacked LHS shape), (N, ...non-stacked RHS shape)
            let mut stack_index_order = Vec::new();
            for idx in cpc.output_indices.iter() {
                if let PairIndexInfo::StackInfo(_) = idx.index_info {
                    stack_index_order.push(idx.index);
                }
            }
            let lhs_reshaped =
                move_stack_indices_to_front(&cpc.lhs_indices, &stack_index_order, &lhs);
            let rhs_reshaped =
                move_stack_indices_to_front(&cpc.rhs_indices, &stack_index_order, &rhs);

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
                output_size: sized_contraction.output_size.clone(),
            };
            let inner_contractor = get_pair_contractor(&new_sized_contraction);

            // (3) for i = 0..N, assign the result of _allused_nostacks_
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
            let temp_result = get_intermediate_result(
                &temp_shape,
                &lhs_reshaped,
                &rhs_reshaped,
                &inner_contractor,
            );
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
}

fn einsum_pair<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    lhs: &ArrayViewD<A>,
    rhs: &ArrayViewD<A>,
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

// API ONLY:
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
    assert_eq!(lhs_axes.len(), rhs_axes.len());
    let lhs_axes_copy: Vec<_> = lhs_axes.iter().map(|x| x.index()).collect();
    let rhs_axes_copy: Vec<_> = rhs_axes.iter().map(|x| x.index()).collect();
    let output_order: Vec<usize> = (0..(lhs.ndim() + rhs.ndim() - 2 * (lhs_axes.len()))).collect();
    let tensordotter = TensordotGeneral::from_shapes_and_axis_numbers(
        &lhs.shape(),
        &rhs.shape(),
        &lhs_axes_copy,
        &rhs_axes_copy,
        &output_order,
    );
    tensordotter.contract_pair(&lhs.view().into_dyn(), &rhs.view().into_dyn())
}

mod wasm_bindings;
pub use wasm_bindings::{
    slow_einsum_with_flattened_operands_as_json_string_as_json,
    validate_and_size_from_shapes_as_string_as_json, validate_as_json,
};

mod slow_versions;
pub use slow_versions::{slow_einsum, slow_einsum_given_sized_contraction};
