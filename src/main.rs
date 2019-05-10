// TODO: Move these to tests
fn _test_parses() {
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

use einsum::einsum;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn rand_array<Sh, D: Dimension>(shape: Sh) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    Sh: ShapeBuilder<Dim = D>,
{
    Array::random(shape, Uniform::new(-5., 5.))
}

fn main() {
    for _ in 0..1000 {
        let m1 = rand_array((60, 40));
        let m2 = rand_array((60, 40));

        let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
        println!("{}", hadamard_product.sum());
    }
}
