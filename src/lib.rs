// #![feature(custom_attribute)]
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

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use ndarray::prelude::*;
use ndarray::{Data, IxDyn, LinalgScalar};

#[derive(Debug)]
pub struct EinsumParse {
    operand_indices: Vec<String>,
    output_indices: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Contraction {
    operand_indices: Vec<String>,
    output_indices: Vec<char>,
    summation_indices: Vec<char>,
}

pub type OutputSize = HashMap<char, usize>;

#[derive(Debug, Serialize)]
pub struct SizedContraction {
    contraction: Contraction,
    output_size: OutputSize,
}

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

fn generate_contraction(parse: &EinsumParse) -> Result<Contraction, &'static str> {
    let mut input_indices = HashMap::new();
    for c in parse.operand_indices.iter().flat_map(|s| s.chars()) {
        *input_indices.entry(c).or_insert(0) += 1;
    }

    let mut unique_indices = Vec::new();
    let mut duplicated_indices = Vec::new();
    for (&c, &n) in input_indices.iter() {
        let dst = if n > 1 {
            &mut duplicated_indices
        } else {
            &mut unique_indices
        };
        dst.push(c);
    }

    let requested_output_indices = match &parse.output_indices {
        Some(s) => s.chars().collect(),
        _ => {
            let mut o = unique_indices.clone();
            o.sort();
            o
        }
    };
    let mut distinct_output_indices = HashMap::new();
    for &c in requested_output_indices.iter() {
        *distinct_output_indices.entry(c).or_insert(0) += 1;
    }
    for (&c, &n) in distinct_output_indices.iter() {
        // No duplicates
        if n > 1 {
            return Err("Requested output has duplicate index");
        }

        // Must be in inputs
        if input_indices.get(&c).is_none() {
            return Err("Requested output contains an index not found in inputs");
        }
    }

    let output_indices = requested_output_indices;
    let mut summation_indices = Vec::new();
    for (&c, _) in input_indices.iter() {
        if distinct_output_indices.get(&c).is_none() {
            summation_indices.push(c);
        }
    }
    summation_indices.sort();

    Ok(Contraction {
        operand_indices: parse.operand_indices.clone(),
        output_indices: output_indices,
        summation_indices: summation_indices,
    })
}

fn parse_einsum_string(input_string: &str) -> Option<EinsumParse> {
    lazy_static! {
        // Unwhitespaced version:
        // ^([a-z]+)((?:,[a-z]+)*)(?:->([a-z]*))?$
        static ref RE: Regex = Regex::new(r"(?x)
            ^
            (?P<first_operand>[a-z]+)
            (?P<more_operands>(?:,[a-z]+)*)
            (?:->(?P<output>[a-z]*))?
            $
            ").unwrap();
    }
    let captures = RE.captures(input_string)?;
    let mut operand_indices = Vec::new();
    let output_indices = captures.name("output").map(|s| String::from(s.as_str()));

    operand_indices.push(String::from(&captures["first_operand"]));
    for s in (&captures["more_operands"]).split(',').skip(1) {
        operand_indices.push(String::from(s));
    }

    Some(EinsumParse {
        operand_indices: operand_indices,
        output_indices: output_indices,
    })
}

pub fn validate(input_string: &str) -> Result<Contraction, &'static str> {
    let p = parse_einsum_string(input_string).ok_or("Invalid string")?;
    generate_contraction(&p)
}

fn get_output_size_from_shapes(
    contraction: &Contraction,
    operand_shapes: &Vec<Vec<usize>>,
) -> Result<OutputSize, &'static str> {
    // Check that len(operand_indices) == len(operands)
    if contraction.operand_indices.len() != operand_shapes.len() {
        return Err("number of operands in contraction does not match number of operands supplied");
    }

    let mut index_lengths: OutputSize = HashMap::new();

    for (indices, operand_shape) in contraction.operand_indices.iter().zip(operand_shapes) {
        // Check that len(operand_indices[i]) == len(operands[i].shape())
        if indices.chars().count() != operand_shape.len() {
            return Err(
                "number of indices in one or more operands does not match dimensions of operand",
            );
        }

        // Check that whenever there are multiple copies of an index,
        // operands[i].shape()[m] == operands[j].shape()[n]
        for (c, &n) in indices.chars().zip(operand_shape) {
            let existing_n = index_lengths.entry(c).or_insert(n);
            if *existing_n != n {
                return Err("repeated index with different size");
            }
        }
    }

    Ok(index_lengths)
}

fn get_operand_shapes<A>(operands: &[&dyn ArrayLike<A>]) -> Vec<Vec<usize>> {
    operands
        .iter()
        .map(|operand| Vec::from(operand.into_dyn_view().shape()))
        .collect()
}

pub fn get_output_size<A>(
    contraction: &Contraction,
    operands: &[&dyn ArrayLike<A>],
) -> Result<HashMap<char, usize>, &'static str> {
    get_output_size_from_shapes(contraction, &get_operand_shapes(operands))
}

fn validate_and_size_from_shapes(
    input_string: &str,
    operand_shapes: &Vec<Vec<usize>>,
) -> Result<SizedContraction, &'static str> {
    let contraction = validate(input_string)?;
    let output_size = get_output_size_from_shapes(&contraction, operand_shapes)?;

    Ok(SizedContraction {
        contraction,
        output_size,
    })
}

pub fn validate_and_size<A>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<SizedContraction, &'static str> {
    validate_and_size_from_shapes(input_string, &get_operand_shapes(operands))
}

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

pub fn generate_classified_singleton_contraction(
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
    let input_chars: Vec<char> = operand_indices[0].chars().collect();

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

pub fn generate_classified_pair_contraction(
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
    let lhs_chars: Vec<char> = operand_indices[0].chars().collect();
    let rhs_chars: Vec<char> = operand_indices[1].chars().collect();

    for (output_position, &c) in output_indices.iter().enumerate() {
        match (
            operand_indices[0].chars().position(|x| x == c),
            operand_indices[1].chars().position(|x| x == c),
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
            operand_indices[0].chars().position(|x| x == c),
            operand_indices[1].chars().position(|x| x == c),
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

fn einsum_singleton_sum_one_index<A>(
    tensor: ArrayD<A>,
    csc: &mut ClassifiedSingletonContraction,
    c: char,
) -> ArrayD<A>
where
    A: LinalgScalar,
{
    // Sums over a single index that appears only once
    let pos = csc.summed_indices[&c].positions[0];

    // (1) Rewrite the ClassifiedSingletonContraction
    for (_, v) in csc.untouched_indices.iter_mut() {
        if v.position > pos {
            (*v).position -= 1;
        }
    }
    csc.summed_indices.remove(&c);
    for (_, v) in csc.summed_indices.iter_mut() {
        for position in v.positions.iter_mut() {
            if *position > pos {
                *position -= 1;
            }
        }
    }
    let all_but_c_indices: Vec<char> = csc
        .input_indices
        .iter()
        .map(|x| x.index)
        .filter(|&x| x != c)
        .collect();
    csc.input_indices = generate_info_vector_for_singleton(
        &all_but_c_indices,
        &csc.untouched_indices,
        &csc.diagonalized_indices,
        &csc.summed_indices,
    );
    let output_indices: Vec<char> = csc.output_indices.iter().map(|x| x.index).collect();
    csc.output_indices = generate_info_vector_for_singleton(
        &output_indices,
        &csc.untouched_indices,
        &csc.diagonalized_indices,
        &csc.summed_indices,
    );

    // (2) Do the sum
    tensor.sum_axis(Axis(pos))
}

fn einsum_singleton_norepeats<A>(
    csc: &ClassifiedSingletonContraction,
    tensor: &ArrayViewD<A>,
) -> ArrayD<A>
where
    A: LinalgScalar,
{
    // Handles the case where it's ijk->ik; just sums
    assert_eq!(csc.diagonalized_indices.len(), 0);
    assert_eq!(
        csc.summed_indices
            .values()
            .filter(|x| x.positions.len() != 1)
            .count(),
        0
    );
    let mut result = tensor.view().into_dyn().into_owned();
    let mut new_csc = (*csc).clone();

    for &c in csc.summed_indices.keys() {
        result = einsum_singleton_sum_one_index(result, &mut new_csc, c);
    }
    assert_eq!(new_csc.summed_indices.len(), 0);
    assert_eq!(new_csc.untouched_indices.len(), csc.output_indices.len());

    // i in the j-th place in the axes sequence means self's i-th axis becomes self.permuted_axes()'s j-th axis
    // If it's iklm->mil, then after summing we'll have ilm
    // we want to make [2, 0, 1]
    let mut permutation = Vec::new();
    for c in csc.output_indices.iter() {
        let pos = new_csc
            .input_indices
            .iter()
            .position(|x| x.index == c.index)
            .unwrap();
        permutation.push(pos);
    }

    result.permuted_axes(permutation)
}

// TODO: Replace this by calculating the right dimensions and strides to use
// TODO: Take a &mut ClassifiedSingletonContraction and mutate it
pub fn diagonalize_singleton<A, S, D>(
    tensor: &ArrayBase<S, D>,
    axes: &[usize],
    destination_axis: usize,
) -> ArrayD<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert!(axes.len() > 0);
    let axis_length = tensor.shape()[axes[0]];
    let slices: Vec<_> = (0..axis_length)
        .map(|i| {
            let mut subview = tensor.view().into_dyn().into_owned();
            let mut foo = Vec::from(axes);
            foo.sort();
            for &j in foo.iter().rev() {
                subview = subview.index_axis(Axis(j), i).into_owned();
            }
            subview.insert_axis(Axis(destination_axis))
        })
        .collect();
    let slice_views: Vec<_> = slices.iter().map(|s| s.view()).collect();
    ndarray::stack(Axis(destination_axis), &slice_views).unwrap()
}

// TODO: Take a &mut ClassifiedSingletonContraction and mutate it instead
// of mutating operand_indices
pub fn diagonalize_singleton_char<A>(
    tensor: &mut ArrayD<A>,
    operand_indices: &mut String,
    repeated_index: char,
) where
    A: LinalgScalar,
{
    let mut new_string = String::from("");
    let mut axes = Vec::new();
    new_string.push(repeated_index);
    for (i, c) in operand_indices.chars().enumerate() {
        if c != repeated_index {
            new_string.push(c);
        } else {
            axes.push(i);
        }
    }
    let new_tensor = diagonalize_singleton(tensor, &axes, 0);

    *tensor = new_tensor;
    *operand_indices = new_string;
}

// TODO: Figure out correct magic with strides and dimensions
// TODO: Take a ClassifiedSingletonContraction instead of a
//       SizedContraction
pub fn einsum_singleton<A, S, D>(
    sized_contraction: &SizedContraction,
    tensor: &ArrayBase<S, D>,
) -> ArrayD<A>
where
    A: LinalgScalar + std::fmt::Debug,
    S: Data<Elem = A>,
    D: Dimension,
{
    // Handles the case where it's iijk->ik; just diagonalization + sums
    assert!(sized_contraction.contraction.operand_indices.len() == 1);
    let mut distinct_elements = HashSet::new();
    let mut repeated_elements = HashSet::new();
    for c in sized_contraction.contraction.operand_indices[0].chars() {
        if distinct_elements.contains(&c) {
            repeated_elements.insert(c);
        } else {
            distinct_elements.insert(c);
        }
    }

    let no_repeated_elements = repeated_elements.len() == 0;

    if no_repeated_elements {
        let csc = generate_classified_singleton_contraction(sized_contraction);
        einsum_singleton_norepeats(&csc, &tensor.view().into_dyn())
    } else {
        let mut operand_indices = sized_contraction.contraction.operand_indices[0].clone();
        let mut modified_tensor = tensor.view().into_dyn().into_owned();

        for &c in repeated_elements.iter() {
            diagonalize_singleton_char(&mut modified_tensor, &mut operand_indices, c);
        }

        // TODO: Just make the new contraction directly
        let mut new_einsum_string = operand_indices.clone();
        new_einsum_string += "->";
        new_einsum_string += &sized_contraction
            .contraction
            .output_indices
            .iter()
            .collect::<String>();
        let new_contraction = validate_and_size(&new_einsum_string, &[&modified_tensor]).unwrap();

        let csc = generate_classified_singleton_contraction(&new_contraction);
        einsum_singleton_norepeats(&csc, &modified_tensor.view())
    }
}

fn tensordot_fixed_order<A, S, S2, D, E>(
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
    last_n: usize,
) -> ArrayD<A>
where
    A: ndarray::LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
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
    let matrix1 = Array::from_shape_vec(lhs.raw_dim(), lhs.iter().cloned().collect())
        .unwrap()
        .into_shape((len_uncontracted_lhs, len_contracted_lhs))
        .unwrap();
    let matrix2 = Array::from_shape_vec(rhs.raw_dim(), rhs.iter().cloned().collect())
        .unwrap()
        .into_shape((len_contracted_rhs, len_uncontracted_rhs))
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

fn einsum_pair_allused_nostacks_classified_deduped_indices<A, S, S2, D, E>(
    classified_pair_contraction: &ClassifiedDedupedPairContraction,
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
) -> ArrayD<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
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

    tensordot(lhs, rhs, &lhs_axes, &rhs_axes).permuted_axes(permutation)
}

pub fn move_stack_indices_to_front<A, S, D>(
    input_indices: &[IndexWithPairInfo],
    tensor: &ArrayBase<S, D>,
) -> ArrayD<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    D: Dimension,
{
    let mut new_order = Vec::new();
    let mut unstacked_order = Vec::new();
    let mut stacked_length = 1;
    let mut unstack_shapes = Vec::new();
    for (i, (idx, &axis_length)) in input_indices.iter().zip(tensor.shape().iter()).enumerate() {
        if let PairIndexInfo::StackInfo(_) = idx.index_info {
            new_order.push(i);
            stacked_length *= axis_length;
        } else {
            unstacked_order.push(i);
            unstack_shapes.push(axis_length);
        }
    }
    new_order.append(&mut unstacked_order);
    let mut output_shape = Vec::new();
    output_shape.push(stacked_length);
    output_shape.append(&mut unstack_shapes);

    let mut permutation = Vec::new();
    for i in 0..(tensor.ndim()) {
        let pos = new_order.iter().position(|&x| x == i).unwrap();
        permutation.push(pos);
    }

    tensor
        .view()
        .into_dyn()
        .into_owned()
        .permuted_axes(permutation)
        .into_shape(IxDyn(&output_shape))
        .unwrap()
}

pub fn einsum_pair_allused_classified_deduped_indices<A, S, S2, D, E>(
    classified_pair_contraction: &ClassifiedDedupedPairContraction,
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
) -> ArrayD<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    // Allowed: abc,bce->ae
    // Allowed: abc,acd -> abd [a in lhs, rhs, and output]
    // Not allowed: abc,bcde -> ae [no d in output]
    // Not allowed: abbc,bce -> ae [repeated b in input]
    // In other words: each index of each tensor is unique,
    // and is either in the other tensor or in the output

    if classified_pair_contraction.stack_indices.len() == 0 {
        einsum_pair_allused_nostacks_classified_deduped_indices(
            classified_pair_contraction,
            lhs,
            rhs,
        )
    } else {
        // What do we have to do?
        // (1) Permute the stack indices to the front of LHS and RHS and
        //     Reshape into (N, ...non-stacked LHS shape), (N, ...non-stacked RHS shape)
        let lhs_reshaped = move_stack_indices_to_front(
            &classified_pair_contraction.lhs_indices,
            &lhs
        );
        let rhs_reshaped = move_stack_indices_to_front(
            &classified_pair_contraction.rhs_indices,
            &rhs
        );

        // (2) Construct the non-stacked ClassifiedDedupedPairContraction
        let unstacked_stack_indices = StackIndexMap::new();
        let mut unstacked_lhs_chars = Vec::new();
        let mut unstacked_rhs_chars = Vec::new();
        let mut unstacked_output_chars = Vec::new();
        for idx in classified_pair_contraction.lhs_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                unstacked_lhs_chars.push(idx.index);
            }
        }
        for idx in classified_pair_contraction.rhs_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                unstacked_rhs_chars.push(idx.index);
            }
        }
        for idx in classified_pair_contraction.output_indices.iter() {
            if let PairIndexInfo::StackInfo(_) = idx.index_info {
            } else {
                unstacked_output_chars.push(idx.index);
            }
        }
        // new_cdpc = ClassifiedDedupedPairContraction{
        //     let lhs_indices = generate_info_vector_for_pair(
        //         &unstacked_lhs_chars,
        //         &unstacked_lhs_chars,
        //         &classified_pair_contraction.outer_indices,
        //         &contracted_indices,
        //     );
        //     let rhs_indices = generate_info_vector_for_pair(
        //         &rhs_chars,
        //         &stack_indices,
        //         &outer_indices,
        //         &contracted_indices,
        //     );
        //     let output_indices = generate_info_vector_for_pair(
        //         &output_indices,
        //         &stack_indices,
        //         &outer_indices,
        //         &contracted_indices,
        //     );
        //
        //     assert_eq!(
        //         output_indices.len(),
        //         stack_indices.len() + outer_indices.len()
        //     );
        //     assert_eq!(summation_indices.len(), contracted_indices.len());
        //
        //     ClassifiedDedupedPairContraction {
        //         generate_info_vector_for_pair(
        //             &lhs_unstacked_chars,
        //             &stack_indices,
        //             &outer_indices,
        //             &contracted_indices,
        //         ),
        //         rhs_indices,
        //         output_indices,
        //         stack_indices,
        //         contracted_indices,
        //         outer_indices,
        //     }
        // }

        // (3) Make unshaped_result = an ArrayD of shape (N, ...non-stacked output shape)
        // (4) for i = 0..N, assign the result of einsum_pair_allused_nostacks_classified_deduped_indices
        //     to unshaped_result[i]
        // (5) Reshape into (...stacked indices, ...non-stacked output shape)
        // (6) Permute into correct order

        lhs.view().into_dyn().into_owned()
    }

}

pub fn einsum_pair_allused_nostacks<A, S, S2, D, E>(
    sized_contraction: &SizedContraction,
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
) -> ArrayD<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    let cpc = generate_classified_pair_contraction(sized_contraction);
    einsum_pair_allused_nostacks_classified_deduped_indices(&cpc, lhs, rhs)
}

//////// Slow stuff below here ////////

fn make_index(indices: &str, bindings: &HashMap<char, usize>) -> IxDyn {
    ////// PYTHON: ///////////////////
    // def make_tuple(
    //     indices,
    //     bindings,
    // ):
    //     return tuple([bindings[x] for x in indices])
    //////////////////////////////////
    let mut v: Vec<usize> = Vec::new();
    for i in indices.chars() {
        v.push(bindings[&i])
    }
    IxDyn(&v)
}

fn partial_einsum_inner_loop<A: LinalgScalar>(
    operands: &[&ArrayViewD<A>],
    operand_indices: &Vec<String>,
    bound_indices: &HashMap<char, usize>,
    axis_lengths: &HashMap<char, usize>,
    free_summation_indices: &[char],
) -> A {
    ////// PYTHON: ///////////////////
    // def partial_einsum_inner_loop(...):
    //     if len(free_summation_indices) == 0:
    //         return np.product([
    //             operand[make_tuple(indices, bound_indices)]
    //             for (operand, indices) in zip(operands, operand_indices)
    //         ])
    //     else:
    //         next_index = free_summation_indices[0]
    //         remaining_indices = free_summation_indices[1:]
    //         partial_sum = 0
    //         for i in range(axis_lengths[next_index]):
    //             partial_sum += partial_einsum_inner_loop(
    //                 operands=operands,
    //                 operand_indices=operand_indices,
    //                 bound_indices={**bound_indices, **{next_index: i}},
    //                 axis_lengths=axis_lengths,
    //                 free_summation_indices=remaining_indices
    //             )
    //         return partial_sum
    //////////////////////////////////
    if free_summation_indices.len() == 0 {
        let mut p = num_traits::identities::one::<A>();
        for (operand, indices) in operands.iter().zip(operand_indices) {
            let index = make_index(&indices, bound_indices);
            p = p * operand[index];
        }
        p
    } else {
        let next_index = free_summation_indices[0];
        let remaining_indices = &free_summation_indices[1..];
        let mut s = num_traits::identities::zero::<A>();
        for i in 0..axis_lengths[&next_index] {
            let mut new_bound_indices = bound_indices.clone();
            new_bound_indices.insert(next_index, i);

            s = s + partial_einsum_inner_loop(
                operands,
                operand_indices,
                &new_bound_indices,
                axis_lengths,
                remaining_indices,
            )
        }
        s
    }
}

fn partial_einsum_outer_loop<A: LinalgScalar>(
    operands: &[&ArrayViewD<A>],
    operand_indices: &Vec<String>,
    bound_indices: &HashMap<char, usize>,
    free_output_indices: &[char],
    axis_lengths: &HashMap<char, usize>,
    summation_indices: &[char],
) -> ArrayD<A> {
    ////// PYTHON: ///////////////////
    // def partial_einsum_outer_loop(...):
    //     if len(free_output_indices) == 0:
    //         return partial_einsum_inner_loop(
    //             operands=operands,
    //             operand_indices=operand_indices,
    //             bound_indices=bound_indices,
    //             axis_lengths=axis_lengths,
    //             free_summation_indices=summation_indices
    //         )
    //     else:
    //         next_index = free_output_indices[0]
    //         remaining_indices = free_output_indices[1:]
    //         return np.array([
    //             partial_einsum_outer_loop(
    //                 operands=operands,
    //                 operand_indices=operand_indices,
    //                 bound_indices={**bound_indices, **{next_index: i}},
    //                 free_output_indices=remaining_indices,
    //                 axis_lengths=axis_lengths,
    //                 summation_indices=summation_indices
    //             )
    //             for i in range(axis_lengths[next_index])
    //         ])
    //////////////////////////////////
    if free_output_indices.len() == 0 {
        arr0(partial_einsum_inner_loop(
            operands,
            operand_indices,
            bound_indices,
            axis_lengths,
            summation_indices,
        ))
        .into_dyn()
    } else {
        let next_index = free_output_indices[0];
        let remaining_indices = &free_output_indices[1..];
        let slices: Vec<_> = (0..axis_lengths[&next_index])
            .map(|i| {
                let mut new_bound_indices = bound_indices.clone();
                new_bound_indices.insert(next_index, i);
                partial_einsum_outer_loop(
                    operands,
                    operand_indices,
                    &new_bound_indices,
                    remaining_indices,
                    axis_lengths,
                    summation_indices,
                )
                .insert_axis(Axis(0))
            })
            .collect();
        let slice_views: Vec<_> = slices.iter().map(|s| s.view()).collect();
        ndarray::stack(Axis(0), &slice_views).unwrap()
    }
}

pub fn slow_einsum_given_sized_contraction<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    operands: &[&dyn ArrayLike<A>],
) -> ArrayD<A> {
    ////// PYTHON: ///////////////////
    // def my_einsum(
    //     contraction,
    //     operands,
    //     axis_lengths,
    // ):
    //     return partial_einsum_outer_loop(
    //         operands=operands,
    //         operand_indices=contraction["operand_indices"],
    //         bound_indices={},
    //         free_output_indices=contraction["output_indices"],
    //         axis_lengths=axis_lengths,
    //         summation_indices=contraction["summation_indices"]
    //     )
    //////////////////////////////////
    let dyn_operands: Vec<ArrayViewD<A>> = operands.iter().map(|x| x.into_dyn_view()).collect();
    // TODO: Figure out why I have to write it this way?!?!
    let operand_refs: Vec<&ArrayViewD<A>> = dyn_operands.iter().map(|x| x).collect();
    let bound_indices: HashMap<char, usize> = HashMap::new();

    partial_einsum_outer_loop(
        &operand_refs,
        &sized_contraction.contraction.operand_indices,
        &bound_indices,
        &sized_contraction.contraction.output_indices,
        &sized_contraction.output_size,
        &sized_contraction.contraction.summation_indices,
    )
}

pub fn slow_einsum<A: LinalgScalar>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<ArrayD<A>, &'static str> {
    let sized_contraction = validate_and_size(input_string, operands)?;
    Ok(slow_einsum_given_sized_contraction(
        &sized_contraction,
        operands,
    ))
}

//////// Versions that accept strings for WASM interop below here ////
#[derive(Debug, Serialize, Deserialize)]
pub struct OperandSizes(Vec<Vec<usize>>);

#[derive(Debug, Serialize, Deserialize)]
pub struct FlattenedOperand<T> {
    pub shape: Vec<usize>,
    pub contents: Vec<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlattenedOperandList<T>(pub Vec<FlattenedOperand<T>>);

fn unflatten_operand<A: LinalgScalar>(
    flattened_operand: &FlattenedOperand<A>,
) -> Result<ArrayD<A>, ndarray::ShapeError> {
    Array::from_shape_vec(
        flattened_operand.shape.clone(),
        flattened_operand.contents.clone(),
    )
}

fn flatten_operand<A: LinalgScalar>(unflattened_operand: &ArrayD<A>) -> FlattenedOperand<A> {
    FlattenedOperand {
        shape: Vec::from(unflattened_operand.shape()),
        contents: unflattened_operand.iter().map(|x| *x).collect::<Vec<A>>(),
    }
}

pub fn validate_and_size_from_shapes_as_string(
    input_string: &str,
    operand_shapes_as_str: &str,
) -> Result<SizedContraction, &'static str> {
    match serde_json::from_str::<OperandSizes>(&operand_shapes_as_str) {
        Err(_) => Err("Error parsing operand shapes into Vec<Vec<usize>>"),
        Ok(OperandSizes(operand_shapes)) => {
            validate_and_size_from_shapes(input_string, &operand_shapes)
        }
    }
}

pub fn slow_einsum_with_flattened_operands<A: LinalgScalar>(
    input_string: &str,
    flattened_operands: &[&FlattenedOperand<A>],
) -> Result<ArrayD<A>, &'static str> {
    let maybe_operands = flattened_operands
        .iter()
        .map(|x| unflatten_operand(*x))
        .collect::<Result<Vec<_>, _>>();
    match maybe_operands {
        Err(_) => Err("Could not unpack one or more flattened operands"),
        Ok(operands) => {
            let mut operand_refs: Vec<&dyn ArrayLike<A>> = Vec::new();
            for operand in operands.iter() {
                operand_refs.push(operand);
            }
            slow_einsum(input_string, &operand_refs)
        }
    }
}

pub fn slow_einsum_with_flattened_operands_as_string_generic<A>(
    input_string: &str,
    flattened_operands_as_string: &str,
) -> Result<ArrayD<A>, &'static str>
where
    A: LinalgScalar + serde::de::DeserializeOwned,
{
    let maybe_flattened_operands =
        serde_json::from_str::<FlattenedOperandList<A>>(flattened_operands_as_string);
    match maybe_flattened_operands {
        Err(_) => Err("Could not parse flattened operands"),
        Ok(FlattenedOperandList(owned_flattened_operands)) => {
            let flattened_operands: Vec<_> = owned_flattened_operands.iter().map(|x| x).collect();
            slow_einsum_with_flattened_operands(input_string, &flattened_operands)
        }
    }
}

pub fn slow_einsum_with_flattened_operands_as_flattened_json_string(
    input_string: &str,
    flattened_operands_as_string: &str,
) -> Result<FlattenedOperand<f64>, &'static str> {
    let maybe_result = slow_einsum_with_flattened_operands_as_string_generic::<f64>(
        input_string,
        flattened_operands_as_string,
    )?;
    Ok(flatten_operand(&maybe_result))
}

////////////////////////// WASM stuff below here ///////////////////////
#[derive(Debug, Serialize)]
pub struct ContractionResult(Result<Contraction, &'static str>);

#[wasm_bindgen(js_name = validateAsJson)]
pub fn validate_as_json(input_string: &str) -> String {
    match serde_json::to_string(&ContractionResult(validate(input_string))) {
        Ok(s) => s,
        _ => String::from("{\"Err\": \"Serialization Error\"}"),
    }
}

#[derive(Debug, Serialize)]
pub struct SizedContractionResult(Result<SizedContraction, &'static str>);

#[wasm_bindgen(js_name = validateAndSizeFromShapesAsStringAsJson)]
pub fn validate_and_size_from_shapes_as_string_as_json(
    input_string: &str,
    operand_shapes: &str,
) -> String {
    match serde_json::to_string(&SizedContractionResult(
        validate_and_size_from_shapes_as_string(input_string, operand_shapes),
    )) {
        Ok(s) => s,
        _ => String::from("{\"Err\": \"Serialization Error\"}"),
    }
}

#[derive(Debug, Serialize)]
pub struct EinsumResult<T>(Result<FlattenedOperand<T>, &'static str>);

#[wasm_bindgen(js_name = slowEinsumAsJson)]
pub fn slow_einsum_with_flattened_operands_as_json_string_as_json(
    input_string: &str,
    flattened_operands_as_string: &str,
) -> String {
    match serde_json::to_string(&EinsumResult(
        slow_einsum_with_flattened_operands_as_flattened_json_string(
            input_string,
            flattened_operands_as_string,
        ),
    )) {
        Ok(s) => s,
        _ => String::from("{\"Err\": \"Serialization Error\"}"),
    }
}
