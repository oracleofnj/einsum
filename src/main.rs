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
        println!("{}", einsum::validate(test_string).to_json().unwrap());
        println!("");
    }
}
