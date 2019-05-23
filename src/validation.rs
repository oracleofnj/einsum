use crate::ArrayLike;
use lazy_static::lazy_static;
use regex::Regex;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct EinsumParse {
    operand_indices: Vec<String>,
    output_indices: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Contraction {
    pub operand_indices: Vec<Vec<char>>,
    pub output_indices: Vec<char>,
    pub summation_indices: Vec<char>,
}

impl Contraction {
    pub fn from_indices(
        operand_indices: &[Vec<char>],
        output_indices: &[char],
    ) -> Result<Self, &'static str> {
        let mut input_char_counts = HashMap::new();
        for &c in operand_indices.iter().flat_map(|operand| operand.iter()) {
            *input_char_counts.entry(c).or_insert(0) += 1;
        }

        let mut distinct_output_indices = HashMap::new();
        for &c in output_indices.iter() {
            *distinct_output_indices.entry(c).or_insert(0) += 1;
        }
        for (&c, &n) in distinct_output_indices.iter() {
            // No duplicates
            if n > 1 {
                return Err("Requested output has duplicate index");
            }

            // Must be in inputs
            if input_char_counts.get(&c).is_none() {
                return Err("Requested output contains an index not found in inputs");
            }
        }

        let mut summation_indices: Vec<char> = input_char_counts
            .keys()
            .filter(|&c| distinct_output_indices.get(c).is_none())
            .cloned()
            .collect();
        summation_indices.sort();

        let cloned_operand_indices: Vec<Vec<char>> = operand_indices.iter().cloned().collect();

        Ok(Contraction {
            operand_indices: cloned_operand_indices,
            output_indices: output_indices.to_vec(),
            summation_indices,
        })
    }
}

pub type OutputSize = HashMap<char, usize>;

#[derive(Debug, Clone, Serialize)]
pub struct SizedContraction {
    pub contraction: Contraction,
    pub output_size: OutputSize,
}

impl SizedContraction {
    pub fn subset(
        &self,
        new_operand_indices: &[Vec<char>],
        new_output_indices: &[char],
    ) -> Result<Self, &'static str> {
        // Make sure all chars in new_operand_indices are in self
        let all_operand_indices: HashSet<char> = new_operand_indices
            .iter()
            .flat_map(|operand| operand.iter())
            .cloned()
            .collect();
        if all_operand_indices
            .iter()
            .any(|c| self.output_size.get(c).is_none())
        {
            return Err("Character found in new_operand_indices but not in self.output_size");
        }

        let new_contraction = Contraction::from_indices(new_operand_indices, new_output_indices)?;
        let new_output_size: OutputSize = self
            .output_size
            .iter()
            .filter(|(&k, _)| all_operand_indices.contains(&k))
            .map(|(&k, &v)| (k, v))
            .collect();

        Ok(SizedContraction {
            contraction: new_contraction,
            output_size: new_output_size,
        })
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
        if n > 1 {
            duplicated_indices.push(c);
        } else {
            unique_indices.push(c);
        };
    }

    // Handle implicit case, e.g. nothing to the right of the arrow
    let requested_output_indices: Vec<char> = match &parse.output_indices {
        Some(s) => s.chars().collect(),
        _ => {
            let mut o = unique_indices.clone();
            o.sort();
            o
        }
    };

    let operand_indices: Vec<Vec<char>> = parse
        .operand_indices
        .iter()
        .map(|x| x.chars().collect::<Vec<char>>())
        .collect();
    Contraction::from_indices(&operand_indices, &requested_output_indices)
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
        if indices.len() != operand_shape.len() {
            return Err(
                "number of indices in one or more operands does not match dimensions of operand",
            );
        }

        // Check that whenever there are multiple copies of an index,
        // operands[i].shape()[m] == operands[j].shape()[n]
        for (&c, &n) in indices.iter().zip(operand_shape) {
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

pub fn validate_and_size_from_shapes(
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
