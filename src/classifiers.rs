use crate::validation::{Contraction, SizedContraction};
use std::collections::HashMap;

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
    pub index: char,
    pub index_info: SingletonIndexInfo,
}

type UntouchedIndexMap = HashMap<char, UntouchedIndex>;
type DiagonalizedIndexMap = HashMap<char, DiagonalizedIndex>;
type SummedIndexMap = HashMap<char, SummedIndex>;

#[derive(Clone, Debug)]
pub struct ClassifiedSingletonContraction {
    pub input_indices: Vec<IndexWithSingletonInfo>,
    pub output_indices: Vec<IndexWithSingletonInfo>,
    pub untouched_indices: UntouchedIndexMap,
    pub diagonalized_indices: DiagonalizedIndexMap,
    pub summed_indices: SummedIndexMap,
}

impl ClassifiedSingletonContraction {
    pub fn new(sc: &SizedContraction) -> Self {
        generate_classified_singleton_contraction(sc)
    }
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
    // assert_eq!(csc.diagonalized_indices.len(), 0);
    // assert_eq!(
    //     csc.summed_indices
    //         .values()
    //         .filter(|x| x.positions.len() != 1)
    //         .count(),
    //     0
    // );

    ClassifiedSingletonContraction {
        input_indices,
        output_indices,
        untouched_indices,
        diagonalized_indices,
        summed_indices,
    }
}
