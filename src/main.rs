pub mod einsum {

    use lazy_static::lazy_static;
    use regex::Regex;
    use std::collections::HashMap;
    #[derive(Debug)]
    pub struct EinsumParse {
        operand_indices: Vec<String>,
        output_indices: Option<String>,
    }

    #[derive(Debug)]
    pub struct Contraction {
        operand_indices: Vec<String>,
        output_indices: Vec<char>,
        summation_indices: Vec<char>,
    }

    pub fn generate_contraction(parse: &EinsumParse) -> Result<Contraction, &str> {
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
}

fn main() {
    for test_string in [
        // Explicit
        "i->",
        "ij->",
        "i->i",
        "ij,ij->ij",
        "ij,ij->",
        "ij,kl->",
        "ij,jk->ik",
        "ijk,jkl,klm->im",
        "ij,jk->ki",
        "ij,ja->ai",
        "ij,ja->ia",
        "ii->i",

        // Implicit
        "ij,k",
        "i",
        "ii",
        "ijj",
        "i,j,klm,nop",
        "ij,jk",
        "ij,ja",

        // Illegal
        "->i",
        "i,",
        "->",
        "i,,,j->k",

        // Legal parse but illegal outputs
        "i,j,k,l,m->p",
        "i,j->ijj",
    ]
    .iter()
    {
        println!("Input string: {}", test_string);
        match einsum::parse_einsum_string(test_string) {
            Some(p) => {
                println!("{:?}", p);
                match einsum::generate_contraction(&p) {
                    Ok(c) => println!("{:?}", c),
                    Err(e) => println!("{:?}", e),
                };
            }
            _ => println!("Invalid string"),
        };
        println!("");
    }
}
