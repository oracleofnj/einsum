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
    let v1 = vec![0, 1, 2];
    println!("{}", v1 == (0..3).collect::<Vec<usize>>());
    let mut max = 0.;
    for _ in 0..1000 {
        let m1 = rand_array((500, 500));
        let m2 = rand_array((500, 500));

        let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
        let hm = hadamard_product.sum();
        max = if max > hm { max } else { hm }
    }
    println!("{}", max);
}
