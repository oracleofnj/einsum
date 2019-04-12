pub mod einsum {
    use regex::Regex;
    use lazy_static::lazy_static;

    #[derive(Debug)]
    pub struct EinsumExpression {
        num_operands: usize,
        operand_indices: Vec<String>,
        output_indices: String
    }

    pub fn parse_einsum_string(input_string: &str) -> Option<EinsumExpression> {
        // Einsum string has the form:
        //
        // ([a..z]*,)*([a-z]*)->([a-z]*)
        lazy_static! {
            static ref RE: Regex = Regex::new(r"(?x)
            ^
            (?P<first_operand>[a-z]+)
            (?P<more_operands>(?:,[a-z]+)*)
            ->
            (?P<output>[a-z]*)
            $
            ").unwrap();
        }

        let captures = RE.captures(input_string)?;
        let mut num_operands = 1;
        let mut operand_indices = Vec::new();

        operand_indices.push(String::from(&captures["first_operand"]));
        for s in (&captures["more_operands"]).split(',').skip(1) {
            operand_indices.push(String::from(s));
            num_operands += 1;
        }

        Some(EinsumExpression {
            num_operands: num_operands,
            operand_indices: operand_indices,
            output_indices: String::from(&captures["output"])
        })
    }
}

fn main() {
    for test_string in [
        "i->",
        "ij->",
        "i->i",
        "ij,ij->ij",
        "ij,ij->",
        "ij,kl->",
        "ij,jk->ik",
        "i,j,k,l,m->p",
        "ijk,jkl,klm->im",
        "->i",
        "i,",
        "ij,k",
        "->",
    ].iter() {
        println!("{}", test_string);
        match einsum::parse_einsum_string(test_string) {
            Some(e) => println!("{:?}", e),
            _ => println!("Invalid string")
        }
    }
}
