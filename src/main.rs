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
//     carrying: bool,
//     ndim: usize,
//     axes: Vec<usize>,
//     shape: Vec<usize>,
//     positions: Vec<usize>,
//     underlying: &'a ArrayViewD<'a, A>,
//     subviews: Vec<ArrayViewD<'a, A>>,
//     axis_iters: Vec<AxisIter<'a, A, IxDyn>>,
// }
//
// // struct MultiAxisIterator<'a, A> {
// //     ndim: usize,
// //     axes: Vec<usize>,
// //     subviews: Vec<ArrayViewD<'a, A>>,
// // }
// //
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
// struct ViewAndIter<'a, A> {
//     axis_iter: AxisIter<'a, A, IxDyn>,
// }
//
// impl<'a, A> ViewAndIter<'a, A> {
//     fn new(base: &'a ArrayViewD<'a, A>) -> ViewAndIter<'a, A> {
//         ViewAndIter {
//             axis_iter: base.axis_iter(Axis(0))
//         }
//     }
// }
//
//
// impl<'a, A> MultiAxisIterator<'a, A> {
//     fn new(base: &'a ArrayViewD<'a, A>, axes: &[usize]) -> MultiAxisIterator<'a, A> {
//         let ndim = axes.len();
//         let axes: Vec<usize> = axes.to_vec();
//         let shape: Vec<usize> = axes
//             .iter()
//             .map(|&x| base.shape().get(x).unwrap())
//             .cloned()
//             .collect();
//         let positions = vec![0; shape.len()];
//
//         let mut subviews = Vec::new();
//         let mut axis_iters = Vec::new();
//
//         for (ax_num, &ax) in axes.iter().enumerate() {
//             let mut subview = base.view();
//             let mut subview2 = base.view();
//             for i in 0..ax_num {
//                 subview = subview.index_axis_move(Axis(0), 0);
//                 subview2 = subview2.index_axis_move(Axis(0), 0);
//             }
//             subviews.push(subview);
//             axis_iters.push(subview2.axis_iter(Axis(0)));
//         }
//
//         MultiAxisIterator {
//             underlying: base,
//             carrying: false,
//             ndim,
//             axes,
//             shape,
//             positions,
//             subviews,
//             axis_iters,
//         }
//     }
// }
//
// // impl<'a, A> MultiAxisIterator<'a, A> {
// //     fn new<'b>(base: &'b ArrayViewD<'b, A>, axes: &[usize]) -> MultiAxisIterator<'a, A>
// //     where 'b: 'a
// //     {
// //         let mut subviews: Vec<ArrayViewD<'a, A>> = Vec::new();
// //         subviews.push(base.view());
// //         let mut outer = base.view();
// //         for &ax in axes {
// //             outer = outer.index_axis(Axis(ax), 0);
// //             subviews.push(outer.view());
// //         }
// //         MultiAxisIterator {
// //             ndim: axes.len(),
// //             axes: axes.to_vec(),
// //             subviews: subviews,
// //         }
// //     }
// // }
// //
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
// // impl<'a, A>  Iterator for MultiAxisIterator<'a, A>
// // {
// //     type Item = ArrayViewD<'a, A>;
// //
// //     fn next(&mut self) -> Option<ArrayViewD<A>> {
// //         let foo = &self.subviews[0];
// //         let bar = foo.view();
// //         Some(bar)
// //     }
// // }
// //
//
// impl<'a, A> Iterator for MultiAxisIterator<'a, A> {
//     type Item = ArrayViewD<'a, A>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.axis_iters[0].next()
//         // if !self.carrying {
//         //     let ret = self.positions.clone();
//         //     self.carrying = true;
//         //     for i in 0..self.ndim {
//         //         let axis = self.ndim - i - 1;
//         //         if self.positions[axis] == self.shape[axis] - 1 {
//         //             self.positions[axis] = 0;
//         //         } else {
//         //             self.positions[axis] += 1;
//         //             self.carrying = false;
//         //             break;
//         //         }
//         //     }
//         //     let foo = self.underlying.index_axis(Axis(0), 0);
//         //     let bar = foo.reborrow();
//         //     Some(bar)
//         // } else {
//         //     None
//         // }
//     }
// }
//
fn main() {
    let m1 = rand_array((2, 3, 4));
    println!("{:?}", einsum("ijk->i", &[&m1]));
    // for p in AxisPositions::new(&[2, 4]) {
    //     println!("{:?}", p)
    // }

    // for _ in 0..1000 {
    //     let m1 = rand_array((60, 40));
    //     let m2 = rand_array((60, 40));
    //
    //     let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
    //     println!("{}", hadamard_product.sum());
    // }
}
