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

// struct AxisPositions {
//     carrying: bool,
//     ndim: usize,
//     shape: Vec<usize>,
//     positions: Vec<usize>,
// }
//
// use ndarray::iter::AxisIter;
//
// struct MultiAxisIterator<'a, A> {
//     ndim: usize,
//     axes: Vec<usize>,
//     subviews: Vec<ArrayViewD<'a, A>>,
// }
//
// impl AxisPositions {
//     fn new(shape: &[usize]) -> AxisPositions {
//         AxisPositions {
//             ndim: shape.len(),
//             shape: shape.to_vec(),
//             positions: vec![0; shape.len()],
//             carrying: false,
//         }
//     }
// }
//
// impl<'a, A> MultiAxisIterator<'a, A> {
//     fn new<'b>(base: &'b ArrayViewD<'b, A>, axes: &[usize]) -> MultiAxisIterator<'a, A>
//     where 'b: 'a
//     {
//         let mut subviews: Vec<ArrayViewD<'a, A>> = Vec::new();
//         subviews.push(base.view());
//         let mut outer = base.view();
//         for &ax in axes {
//             outer = outer.index_axis(Axis(ax), 0);
//             subviews.push(outer.view());
//         }
//         MultiAxisIterator {
//             ndim: axes.len(),
//             axes: axes.to_vec(),
//             subviews: subviews,
//         }
//     }
// }
//
// impl Iterator for AxisPositions {
//     type Item = Vec<usize>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if !self.carrying {
//             let ret = self.positions.clone();
//             self.carrying = true;
//             for i in 0..self.ndim {
//                 let axis = self.ndim - i - 1;
//                 if self.positions[axis] == self.shape[axis] - 1 {
//                     self.positions[axis] = 0;
//                 } else {
//                     self.positions[axis] += 1;
//                     self.carrying = false;
//                     break;
//                 }
//             }
//             Some(ret)
//         } else {
//             None
//         }
//     }
// }
//
// impl<'a, A>  Iterator for MultiAxisIterator<'a, A>
// {
//     type Item = ArrayViewD<'a, A>;
//
//     fn next(&mut self) -> Option<ArrayViewD<A>> {
//         let foo = &self.subviews[0];
//         let bar = foo.view();
//         Some(bar)
//     }
// }
//

fn main() {
    let mut max = 0.;
    for _ in 0..10000 {
        let m1 = rand_array((60, 40));
        let m2 = rand_array((60, 40));

        let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
        let hm = hadamard_product.sum();
        max = if max > hm { max } else { hm }
    }
    println!("{}", max);
}
