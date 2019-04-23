use ndarray::prelude::*;

fn test_parses() {
    for test_string in &vec![
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
    ] {
        println!("Input string: {}", test_string);
        println!("{}", einsum::validate_as_json(test_string));
        println!("");
    }
}

fn main() {
    test_parses();

    let r = vec![10, 3, 4];
    let a = Array::<u8, _>::zeros(r);

    let p = vec![4, 5];
    let b = Array::<u8, _>::zeros(p);

    let c = einsum::validate("cij,jk->cik").unwrap();
    println!("{:?}", einsum::get_output_size(&c, &[&a, &b]));

    let c = einsum::validate("ii->i").unwrap();
    println!("{:?}", einsum::get_output_size(&c, &[&b]));

    let b = arr2(&[[4, 5], [2, 2]]);
    println!("{:?}", einsum::get_output_size(&c, &[&b]));

}
