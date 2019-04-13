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

    pub fn generate_contraction(parse: &EinsumParse) -> Option<Contraction> {
        let mut input_indices = HashMap::new();
        for s in &parse.operand_indices {
            for c in s.chars() {
                *input_indices.entry(c).or_insert(0) += 1;
            }
        }

        let mut unique_indices = Vec::new();
        let mut duplicated_indices = Vec::new();
        for (&c, &n) in input_indices.iter() {
            if n > 1 {
                duplicated_indices.push(c);
            } else {
                unique_indices.push(c);
            }
        }

        let output_indices = match &parse.output_indices {
            Some(s) => s.chars().collect(),
            _ => {
                let mut o = unique_indices.clone();
                o.sort();
                o
            }
        };

        println!("Output indices: {:?}", &output_indices);

        Some(Contraction {
            operand_indices: vec![String::from("foo")],
            output_indices: output_indices,
            summation_indices: vec!['b'],
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
        "i,j,k,l,m->p",
        "ijk,jkl,klm->im",
        "ij,jk->ki",
        "ij,ja->ai",
        "ij,ja->ia",

        // Implicit
        "ij,k",
        "i",
        "iii",
        "i,j,klm,nop",
        "ij,jk",
        "ij,ja",

        // Illegal
        "->i",
        "i,",
        "->",
        "i,,,j->k",
    ]
    .iter()
    {
        println!("Input string: {}", test_string);
        match einsum::parse_einsum_string(test_string) {
            Some(e) => {
                println!("{:?}", e);
                einsum::generate_contraction(&e);
            }
            _ => println!("Invalid string"),
        };
        println!("");
    }
}
