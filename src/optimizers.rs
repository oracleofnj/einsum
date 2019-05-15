use crate::{validate, SizedContraction};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct OperandNumPair {
    pub lhs: usize,
    pub rhs: usize,
}

#[derive(Debug)]
pub struct FirstStep {
    pub sized_contraction: SizedContraction,
    pub operand_nums: Option<OperandNumPair>,
}

#[derive(Debug)]
pub struct IntermediateStep {
    pub sized_contraction: SizedContraction,
    pub rhs_num: usize,
}

#[derive(Debug)]
pub struct EinsumPath {
    pub first_step: FirstStep,
    pub remaining_steps: Vec<IntermediateStep>,
}

#[derive(Debug)]
pub enum OptimizationMethod {
    Naive,
    Greedy,
    Optimal,
    Branch,
}

fn get_remaining_indices(operand_indices: &[Vec<char>], output_indices: &[char]) -> HashSet<char> {
    let mut result: HashSet<char> = HashSet::new();
    for &c in operand_indices.iter().flat_map(|s| s.iter()) {
        result.insert(c);
    }
    for &c in output_indices.iter() {
        result.insert(c);
    }
    result
}

fn get_existing_indices(lhs_indices: &HashSet<char>, rhs_indices: &[char]) -> HashSet<char> {
    let mut result: HashSet<char> = lhs_indices.clone();
    for &c in rhs_indices.iter() {
        result.insert(c);
    }
    result
}

fn generate_naive_path(sized_contraction: &SizedContraction) -> EinsumPath {
    match sized_contraction.contraction.operand_indices.len() {
        1 => {
            let first_step = FirstStep {
                sized_contraction: sized_contraction.clone(),
                operand_nums: None,
            };
            EinsumPath {
                first_step,
                remaining_steps: Vec::new(),
            }
        }
        2 => {
            let first_step = FirstStep {
                sized_contraction: sized_contraction.clone(),
                operand_nums: Some(OperandNumPair { lhs: 0, rhs: 1 }),
            };
            EinsumPath {
                first_step,
                remaining_steps: Vec::new(),
            }
        }
        _ => {
            let mut lhs_indices: HashSet<char> = sized_contraction.contraction.operand_indices[0]
                .iter()
                .cloned()
                .collect();
            let rhs_indices = &sized_contraction.contraction.operand_indices[1];
            let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
            let remaining_indices = get_remaining_indices(
                &sized_contraction.contraction.operand_indices[2..],
                &sized_contraction.contraction.output_indices,
            );
            let mut output_indices: Vec<char> = existing_indices
                .intersection(&remaining_indices)
                .cloned()
                .collect();
            let mut output_str: String = output_indices.iter().collect();
            let lhs_str: String = sized_contraction.contraction.operand_indices[0]
                .iter()
                .collect();
            let rhs_str: String = rhs_indices.iter().collect();
            let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
            let contraction = validate(&einsum_str).unwrap();
            let mut output_size: HashMap<char, usize> = sized_contraction.output_size.clone();
            output_size.retain(|k, _| existing_indices.contains(k));
            let sc = SizedContraction {
                contraction,
                output_size,
            };
            let first_step = FirstStep {
                sized_contraction: sc,
                operand_nums: Some(OperandNumPair { lhs: 0, rhs: 1 }),
            };
            let mut remaining_steps = Vec::new();
            for (i, rhs_indices) in sized_contraction.contraction.operand_indices[2..]
                .iter()
                .enumerate()
            {
                let idx_of_rhs = i + 2;
                lhs_indices = output_indices.iter().cloned().collect();
                let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
                let remaining_indices = get_remaining_indices(
                    &sized_contraction.contraction.operand_indices[idx_of_rhs..],
                    &sized_contraction.contraction.output_indices,
                );
                let lhs_str = output_str.clone();
                if idx_of_rhs == sized_contraction.contraction.operand_indices.len() {
                    // Used up all of operands
                    output_indices = existing_indices
                        .intersection(&remaining_indices)
                        .cloned()
                        .collect();
                } else {
                    output_indices = sized_contraction.contraction.output_indices.clone();
                }
                output_str = output_indices.iter().collect();
                let rhs_str: String = rhs_indices.iter().collect();
                let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
                let contraction = validate(&einsum_str).unwrap();
                let mut output_size: HashMap<char, usize> = sized_contraction.output_size.clone();
                output_size.retain(|k, _| existing_indices.contains(k));
                let sc = SizedContraction {
                    contraction,
                    output_size,
                };
                remaining_steps.push(IntermediateStep {
                    sized_contraction: sc,
                    rhs_num: idx_of_rhs,
                });
            }

            EinsumPath {
                first_step,
                remaining_steps,
            }
        }
    }
}

pub fn generate_optimized_path(
    sized_contraction: &SizedContraction,
    strategy: OptimizationMethod,
) -> EinsumPath {
    match strategy {
        OptimizationMethod::Naive => generate_naive_path(sized_contraction),
        _ => panic!("Unsupported optimization method"),
    }
}
