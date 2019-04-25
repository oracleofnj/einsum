#![feature(custom_attribute)]

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Result as SerdeResult;
use wasm_bindgen::prelude::*;

use ndarray::prelude::*;
use ndarray::Data;

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

#[derive(Debug, Serialize, Deserialize)]
pub struct OperandSizes(Vec<Vec<usize>>);

impl FromStr for OperandSizes {
    type Err = serde_json::error::Error;

    fn from_str(s: &str) -> Result<OperandSizes, Self::Err> {
        serde_json::from_str(s)
    }
}

pub fn generate_contraction(parse: &EinsumParse) -> Result<Contraction, &'static str> {
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

pub fn parse_einsum_string(input_string: &str) -> Option<EinsumParse> {
    lazy_static! {
        // Unwhitespaced version:
        // ^([a-z]+)((?:,[a-z]+)*)(?:->([a-z]*))?$

        // Avoiding unwrap() here in a fruitless attempt to shrink
        // generated wasm size
        static ref RE: Regex = {
            match Regex::new(r"(?x)
            ^
            (?P<first_operand>[a-z]+)
            (?P<more_operands>(?:,[a-z]+)*)
            (?:->(?P<output>[a-z]*))?
            $
            ") {
                Ok(r) => r,
                _ => std::process::abort()
            }
        };
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

pub fn get_output_size_from_shapes(
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

pub fn get_operand_shapes<A>(operands: &[&dyn ArrayLike<A>]) -> Vec<Vec<usize>> {
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

pub fn validate_and_size_from_shapes_as_string(
    input_string: &str,
    operand_shapes_as_str: &str,
) -> Result<SizedContraction, &'static str> {
    match operand_shapes_as_str.parse::<OperandSizes>() {
        Err(_) => Err("Error parsing operand shapes into Vec<Vec<usize>>"),
        Ok(OperandSizes(operand_shapes)) => {
            validate_and_size_from_shapes(input_string, &operand_shapes)
        }
    }
}

pub fn validate_and_size<A>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<SizedContraction, &'static str> {
    validate_and_size_from_shapes(input_string, &get_operand_shapes(operands))
}

#[derive(Debug, Serialize)]
pub struct ContractionResult(Result<Contraction, &'static str>);

impl ContractionResult {
    pub fn to_json(&self) -> SerdeResult<String> {
        serde_json::to_string(&self)
    }
}

#[derive(Debug, Serialize)]
pub struct SizedContractionResult(Result<SizedContraction, &'static str>);

impl SizedContractionResult {
    pub fn to_json(&self) -> SerdeResult<String> {
        serde_json::to_string(&self)
    }
}

#[wasm_bindgen(js_name = validateAsJson)]
pub fn validate_as_json(input_string: &str) -> String {
    match ContractionResult(validate(input_string)).to_json() {
        Ok(s) => s,
        _ => String::from("{\"Err\": \"Serialization Error\"}"),
    }
}

#[wasm_bindgen(js_name = validateAndSizeFromShapesAsStringAsJson)]
pub fn validate_and_size_from_shapes_as_string_as_json(
    input_string: &str,
    operand_shapes: &str,
) -> String {
    match SizedContractionResult(validate_and_size_from_shapes_as_string(
        input_string,
        operand_shapes,
    ))
    .to_json()
    {
        Ok(s) => s,
        _ => String::from("{\"Err\": \"Serialization Error\"}"),
    }
}
