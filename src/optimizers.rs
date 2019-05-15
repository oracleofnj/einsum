use crate::{validate, Contraction, OutputSize, SizedContraction};
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

fn generate_permuted_contraction(
    sized_contraction: &SizedContraction,
    tensor_order: &[usize],
) -> SizedContraction {
    assert_eq!(
        sized_contraction.contraction.operand_indices.len(),
        tensor_order.len()
    );
    let mut new_operand_indices = Vec::new();
    for &i in tensor_order {
        new_operand_indices.push(sized_contraction.contraction.operand_indices[i].clone());
    }
    let new_contraction = Contraction {
        operand_indices: new_operand_indices,
        output_indices: sized_contraction.contraction.output_indices.clone(),
        summation_indices: sized_contraction.contraction.summation_indices.clone(),
    };
    SizedContraction {
        contraction: new_contraction,
        output_size: sized_contraction.output_size.clone(),
    }
}

fn generate_sized_contraction_pair(
    lhs_operand_indices: &[char],
    rhs_operand_indices: &[char],
    output_indices: &[char],
    full_output_size: &OutputSize,
) -> SizedContraction {
    let lhs_str: String = lhs_operand_indices.iter().collect();
    let rhs_str: String = rhs_operand_indices.iter().collect();
    let output_str: String = output_indices.iter().collect();
    let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
    let contraction = validate(&einsum_str).unwrap();
    let output_size: OutputSize = full_output_size
        .iter()
        .filter(|(&k, _)| {
            if let Some(_) = output_indices.iter().position(|&c| c == k) {
                true
            } else {
                false
            }
        })
        .map(|(&k, &v)| (k, v))
        .collect();

    SizedContraction {
        contraction,
        output_size,
    }
}

fn generate_path(sized_contraction: &SizedContraction, tensor_order: &[usize]) -> EinsumPath {
    let permuted_contraction = generate_permuted_contraction(sized_contraction, tensor_order);
    match permuted_contraction.contraction.operand_indices.len() {
        1 => {
            let first_step = FirstStep {
                sized_contraction: permuted_contraction.clone(),
                operand_nums: None,
            };
            EinsumPath {
                first_step,
                remaining_steps: Vec::new(),
            }
        }
        2 => {
            let sc = generate_sized_contraction_pair(
                &permuted_contraction.contraction.operand_indices[0],
                &permuted_contraction.contraction.operand_indices[1],
                &permuted_contraction.contraction.output_indices,
                &permuted_contraction.output_size,
            );
            let first_step = FirstStep {
                sized_contraction: sc,
                operand_nums: Some(OperandNumPair {
                    lhs: tensor_order[0],
                    rhs: tensor_order[1],
                }),
            };
            EinsumPath {
                first_step,
                remaining_steps: Vec::new(),
            }
        }
        _ => {
            let mut lhs_indices: HashSet<char> = permuted_contraction.contraction.operand_indices
                [0]
            .iter()
            .cloned()
            .collect();
            let rhs_indices = &permuted_contraction.contraction.operand_indices[1];
            let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
            let remaining_indices = get_remaining_indices(
                &permuted_contraction.contraction.operand_indices[2..],
                &permuted_contraction.contraction.output_indices,
            );
            let mut output_indices: Vec<char> = existing_indices
                .intersection(&remaining_indices)
                .cloned()
                .collect();
            let mut output_str: String = output_indices.iter().collect();
            let lhs_str: String = permuted_contraction.contraction.operand_indices[0]
                .iter()
                .collect();
            let rhs_str: String = rhs_indices.iter().collect();
            let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
            let contraction = validate(&einsum_str).unwrap();
            let mut output_size: HashMap<char, usize> = permuted_contraction.output_size.clone();
            output_size.retain(|k, _| existing_indices.contains(k));
            let sc = SizedContraction {
                contraction,
                output_size,
            };
            let first_step = FirstStep {
                sized_contraction: sc,
                operand_nums: Some(OperandNumPair {
                    lhs: tensor_order[0],
                    rhs: tensor_order[1],
                }),
            };
            let mut remaining_steps = Vec::new();
            for (i, rhs_indices) in permuted_contraction.contraction.operand_indices[2..]
                .iter()
                .enumerate()
            {
                let idx_of_rhs = i + 2;
                lhs_indices = output_indices.iter().cloned().collect();
                let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
                let remaining_indices = get_remaining_indices(
                    &permuted_contraction.contraction.operand_indices[idx_of_rhs..],
                    &permuted_contraction.contraction.output_indices,
                );
                let lhs_str = output_str.clone();
                if idx_of_rhs == permuted_contraction.contraction.operand_indices.len() {
                    // Used up all of operands
                    output_indices = existing_indices
                        .intersection(&remaining_indices)
                        .cloned()
                        .collect();
                } else {
                    output_indices = permuted_contraction.contraction.output_indices.clone();
                }
                output_str = output_indices.iter().collect();
                let rhs_str: String = rhs_indices.iter().collect();
                let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
                let contraction = validate(&einsum_str).unwrap();
                let mut output_size: HashMap<char, usize> =
                    permuted_contraction.output_size.clone();
                output_size.retain(|k, _| existing_indices.contains(k));
                let sc = SizedContraction {
                    contraction,
                    output_size,
                };
                remaining_steps.push(IntermediateStep {
                    sized_contraction: sc,
                    rhs_num: tensor_order[idx_of_rhs],
                });
            }

            EinsumPath {
                first_step,
                remaining_steps,
            }
        }
    }
}

fn naive_order(sized_contraction: &SizedContraction) -> Vec<usize> {
    (0..sized_contraction.contraction.operand_indices.len()).collect()
}

pub fn generate_optimized_path(
    sized_contraction: &SizedContraction,
    strategy: OptimizationMethod,
) -> EinsumPath {
    let tensor_order: Vec<usize> = match strategy {
        OptimizationMethod::Naive => naive_order(sized_contraction),
        _ => panic!("Unsupported optimization method"),
    };
    generate_path(sized_contraction, &tensor_order)
}
