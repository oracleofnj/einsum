pub mod einsum {
    use regex::Regex;
    use lazy_static::lazy_static;

    #[derive(Debug)]
    pub struct EinsumParse {
        num_operands: usize,
        operand_indices: Vec<String>,
        output_indices: Option<String>
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
        let mut num_operands = 1;
        let mut operand_indices = Vec::new();
        let output_indices = captures.name("output").map(|s|  {
            String::from(s.as_str())
        });

        operand_indices.push(String::from(&captures["first_operand"]));
        for s in (&captures["more_operands"]).split(',').skip(1) {
            operand_indices.push(String::from(s));
            num_operands += 1;
        }

        Some(EinsumParse {
            num_operands: num_operands,
            operand_indices: operand_indices,
            output_indices: output_indices
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

        // Implicit
        "ij,k",
        "i",
        "iii",
        "i,j,klm,nop",

        // Illegal
        "->i",
        "i,",
        "->",
        "i,,,j->k"
    ].iter() {
        println!("{}", test_string);
        match einsum::parse_einsum_string(test_string) {
            Some(e) => println!("{:?}", e),
            _ => println!("Invalid string")
        }
    }
}
