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
    let s = arr2(&[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]);
    let sl = s.as_slice_memory_order().unwrap();
    let t = ArrayView::from_shape(IxDyn(&[3]).strides(IxDyn(&[4])), &sl).unwrap();
    let x;
    x = 5;
    println!("{:?} {}", t, x);
    let mut max = 0.;
    for _ in 0..10 {
        let m1 = rand_array((500, 500));
        let m2 = rand_array((500, 500));

        let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
        let hm = hadamard_product.sum();
        max = if max > hm { max } else { hm }
    }
    println!("{}", max);
}
