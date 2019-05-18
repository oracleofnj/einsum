use einsum::this_test_is_annoying;
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
    this_test_is_annoying();
    let mut max = 0.;
    for _ in 0..10 {
        let m1 = rand_array((50, 50));
        let m2 = rand_array((50, 50));

        let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
        let hm = hadamard_product.sum();
        max = if max > hm { max } else { hm }
    }
    println!("{}", max);
}
