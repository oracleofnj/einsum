use crate::{Contraction, SizedContraction};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StackIndex {
    // Which dimension of the LHS tensor does this index correspond to
    pub lhs_position: usize,
    // Which dimension of the RHS tensor does this index correspond to
    pub rhs_position: usize,
    // Which dimension in the output does this index get assigned to
    pub output_position: usize,
}

#[derive(Clone, Debug)]
pub struct ContractedIndex {
    // Which dimension of the LHS tensor does this index correspond to
    pub lhs_position: usize,
    // Which dimension of the RHS tensor does this index correspond to
    pub rhs_position: usize,
}

#[derive(Clone, Debug)]
pub enum OuterIndexPosition {
    LHS(usize),
    RHS(usize),
}

#[derive(Clone, Debug)]
pub struct OuterIndex {
    // Which dimension of the input tensor does this index correspond to
    pub input_position: OuterIndexPosition,
    // Which dimension of the output tensor does this index get assigned to
    pub output_position: usize,
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
    pub index: char,
    pub index_info: PairIndexInfo,
}

type StackIndexMap = HashMap<char, StackIndex>;
type ContractedIndexMap = HashMap<char, ContractedIndex>;
type OuterIndexMap = HashMap<char, OuterIndex>;

#[derive(Debug)]
pub struct ClassifiedDedupedPairContraction {
    pub lhs_indices: Vec<IndexWithPairInfo>,
    pub rhs_indices: Vec<IndexWithPairInfo>,
    pub output_indices: Vec<IndexWithPairInfo>,
    pub stack_indices: StackIndexMap,
    pub contracted_indices: ContractedIndexMap,
    pub outer_indices: OuterIndexMap,
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
