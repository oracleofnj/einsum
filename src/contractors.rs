use crate::{Contraction, SizedContraction};
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use std::collections::HashMap;

pub trait SingletonContractor<'a, A> {
    fn contract_singleton(&self, tensor: &'a ArrayViewD<'a, A>) -> ArrayD<A>
    where
        A: Clone + LinalgScalar;
}

pub trait PairContractor<'a, A> {
    fn contract_pair(&self, lhs: &'a ArrayViewD<'a, A>, rhs: &'a ArrayViewD<'a, A>) -> ArrayD<A>
    where
        A: Clone + LinalgScalar;
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

fn move_output_indices_to_front<'a, A: LinalgScalar>(
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

impl<'a, A> SingletonContractor<'a, A> for ClassifiedSingletonContraction {
    fn contract_singleton(&self, tensor: &'a ArrayViewD<'a, A>) -> ArrayD<A>
    where
        A: Clone + LinalgScalar,
    {
        einsum_singleton_norepeats(&self, &tensor)
    }
}
