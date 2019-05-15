use crate::{validate, Contraction, OutputSize, SizedContraction};
use std::collections::HashSet;

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
    Reverse, // Just for testing
    Greedy,
    Optimal,
    Branch,
}

fn get_remaining_indices(operand_indices: &[Vec<char>], output_indices: &[char]) -> HashSet<char> {
    // Returns a set of all the indices in any of the remaining operands or in the output
    let mut result: HashSet<char> = HashSet::new();
    for &c in operand_indices.iter().flat_map(|s| s.iter()) {
        result.insert(c);
    }
    for &c in output_indices.iter() {
        result.insert(c);
    }
    result
}

fn get_existing_indices(lhs_indices: &[char], rhs_indices: &[char]) -> HashSet<char> {
    // Returns a set of all the indices in the LHS or the RHS
    let mut result: HashSet<char> = lhs_indices.iter().cloned().collect();
    for &c in rhs_indices.iter() {
        result.insert(c);
    }
    result
}

fn generate_permuted_contraction(
    sized_contraction: &SizedContraction,
    tensor_order: &[usize],
) -> SizedContraction {
    // Reorder the operands of the SizedContraction and clone everything else
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
    // Generate a new SizedContraction for a single pair from the larger SizedContraction.
    // This delegates the details back to validate() by creating a string rather than
    // duplicate the code. Potentially inefficient if called in a loop, but is only
    // executed once for each pair.

    let lhs_str: String = lhs_operand_indices.iter().collect();
    let rhs_str: String = rhs_operand_indices.iter().collect();
    let output_str: String = output_indices.iter().collect();

    let einsum_str = format!("{},{}->{}", &lhs_str, &rhs_str, &output_str);
    let contraction = validate(&einsum_str).unwrap();

    // Only keep the output sizes that are actually used in this contraction
    let mut index_set: HashSet<char> = lhs_operand_indices.iter().cloned().collect();
    for &c in rhs_operand_indices {
        index_set.insert(c);
    }
    for &c in output_indices {
        index_set.insert(c);
    }

    let output_size: OutputSize = full_output_size
        .iter()
        .filter(|(k, _)| index_set.contains(k))
        .map(|(&k, &v)| (k, v))
        .collect();

    SizedContraction {
        contraction,
        output_size,
    }
}

fn generate_path(sized_contraction: &SizedContraction, tensor_order: &[usize]) -> EinsumPath {
    // Generate the actual path consisting of all the mini-contractions.

    // Make a reordered full SizedContraction in the order specified by the called
    let permuted_contraction = generate_permuted_contraction(sized_contraction, tensor_order);

    match permuted_contraction.contraction.operand_indices.len() {
        1 => {
            // If there's only one input tensor, make a single-step path consisting of a
            // singleton contraction (operand_nums = None).
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
            // If there's exactly two input tensors, make a single-step path consisting
            // of a pair contraction (operand_nums = Some(OperandNumPair)).
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
            // If there's three or more input tensors, we have some work to do.

            // optional_first_step gets set on the first iteration of the loop;
            // making it an Option is to prevent compiler from complaining
            // about possibly uninitialized value
            let mut optional_first_step = None;
            let mut remaining_steps = Vec::new();
            // In the main body of the loop, output_indices will contain the result of the prior pair
            // contraction. Initialize it to the elements of the first LHS tensor so that we can
            // clone it on the first go-around as well as all the later ones.
            let mut output_indices = permuted_contraction.contraction.operand_indices[0].clone();

            for idx_of_lhs in 0..(permuted_contraction.contraction.operand_indices.len() - 1) {
                // lhs_indices is either the first tensor (on the first iteration of the loop)
                // or the output from the previous step.
                let lhs_indices = output_indices.clone();

                // rhs_indices is always the next tensor.
                let idx_of_rhs = idx_of_lhs + 1;
                let rhs_indices = &permuted_contraction.contraction.operand_indices[idx_of_rhs];

                // existing_indices and remaining_indices are only needed to figure out
                // what output_indices will be for this step.
                //
                // existing_indices consists of the indices in either the LHS or the RHS tensor
                // for this step.
                //
                // remaining_indices consists of the indices in all the elements after the RHS
                // tensor or in the outputs.
                //
                // The output indices we want is the intersection of the two (unless this is
                // the RHS is the last operand, in which case it's just the output indices).
                //
                // For example, say the string is "ij,jk,kl,lm->im".
                // First iteration:
                //      lhs = [i,j]
                //      rhs = [j,k]
                //      existing = {i,j,k}
                //      remaining = {k,l,m,i} (the i is used in the final output so we need to
                //      keep it around)
                //      output = {i,k}
                //      Mini-contraction: ij,jk->ik
                // Second iteration:
                //      lhs = [i,k]
                //      rhs = [k,l]
                //      existing = {i,k,l}
                //      remaining = {l,m,i}
                //      output = {i,l}
                //      Mini-contraction: ik,kl->il
                // Third (and final) iteration:
                //      lhs = [i,l]
                //      rhs = [l,m]
                //      (Short-circuit) output = {i,m}
                //      Mini-contraction: il,lm->im
                output_indices =
                    if idx_of_rhs == (permuted_contraction.contraction.operand_indices.len() - 1) {
                        // Used up all the operands; just return output
                        permuted_contraction.contraction.output_indices.clone()
                    } else {
                        let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
                        let remaining_indices = get_remaining_indices(
                            &permuted_contraction.contraction.operand_indices[(idx_of_rhs + 1)..],
                            &permuted_contraction.contraction.output_indices,
                        );
                        existing_indices
                            .intersection(&remaining_indices)
                            .cloned()
                            .collect()
                    };

                // Phew, now make the mini-contraction.
                let sc = generate_sized_contraction_pair(
                    &lhs_indices,
                    &rhs_indices,
                    &output_indices,
                    &permuted_contraction.output_size,
                );

                if idx_of_lhs == 0 {
                    optional_first_step = Some(FirstStep {
                        sized_contraction: sc,
                        operand_nums: Some(OperandNumPair {
                            lhs: tensor_order[idx_of_lhs], // tensor_order[0]
                            rhs: tensor_order[idx_of_rhs], // tensor_order[1]
                        }),
                    });
                } else {
                    remaining_steps.push(IntermediateStep {
                        sized_contraction: sc,
                        rhs_num: tensor_order[idx_of_rhs],
                    });
                }
            }

            match optional_first_step {
                // Shouldn't be possible to still be None since this
                // gets set on the first pass through the loop.
                Some(first_step) => EinsumPath {
                    first_step,
                    remaining_steps,
                },
                None => panic!(),
            }
        }
    }
}

fn naive_order(sized_contraction: &SizedContraction) -> Vec<usize> {
    (0..sized_contraction.contraction.operand_indices.len()).collect()
}

fn reverse_order(sized_contraction: &SizedContraction) -> Vec<usize> {
    (0..sized_contraction.contraction.operand_indices.len())
        .rev()
        .collect()
}

// TODO: Maybe this should take a function pointer from &SizedContraction -> Vec<usize>?
pub fn generate_optimized_path(
    sized_contraction: &SizedContraction,
    strategy: OptimizationMethod,
) -> EinsumPath {
    let tensor_order = match strategy {
        OptimizationMethod::Naive => naive_order(sized_contraction),
        OptimizationMethod::Reverse => reverse_order(sized_contraction),
        _ => panic!("Unsupported optimization method"),
    };
    generate_path(sized_contraction, &tensor_order)
}
