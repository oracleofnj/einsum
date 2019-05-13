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

fn get_new_axis_order(axes: &[usize]) -> Vec<usize> {
    let new_axis_order: Vec<usize> = axes
        .iter()
        .enumerate()
        .map(|(i, &v)| v - axes[0..i].iter().filter(|&&x| x < v).count())
        .collect();
    new_axis_order
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
struct MultiAxisIterator<'a, A> {
    carrying: bool,
    ndim: usize,
    // axes: Vec<usize>,
    renumbered_axes: Vec<usize>,
    shape: Vec<usize>,
    positions: Vec<usize>,
    underlying: &'a ArrayViewD<'a, A>,
    // subviews: Vec<ArrayViewD<'a, A>>,
}

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
impl<'a, A> MultiAxisIterator<'a, A> {
    fn new(base: &'a ArrayViewD<'a, A>, axes: &[usize]) -> MultiAxisIterator<'a, A> {
        let ndim = axes.len();
        // let axes: Vec<usize> = axes.to_vec();
        let renumbered_axes = get_new_axis_order(&axes);
        let shape: Vec<usize> = axes
            .iter()
            .map(|&x| base.shape().get(x).unwrap())
            .cloned()
            .collect();
        let positions = vec![0; shape.len()];

        // let mut subviews = Vec::new();
        // let mut axis_iters = Vec::new();
        //
        // for (ax_num, &ax) in axes.iter().enumerate() {
        //     let mut subview = base.view();
        //     for i in 0..ax_num {
        //         subview = subview.index_axis_move(Axis(0), 0);
        //     }
        //     subviews.push(subview);
        // }

        MultiAxisIterator {
            underlying: base,
            carrying: false,
            ndim,
            // axes,
            renumbered_axes,
            shape,
            positions,
            // subviews,
        }
    }
}

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

impl<'a, A> Iterator for MultiAxisIterator<'a, A> {
    type Item = ArrayViewD<'a, A>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.carrying {
            let ret = self.positions.clone();
            self.carrying = true;
            for i in 0..self.ndim {
                let axis = self.ndim - i - 1;
                if self.positions[axis] == self.shape[axis] - 1 {
                    self.positions[axis] = 0;
                } else {
                    self.positions[axis] += 1;
                    self.carrying = false;
                    break;
                }
            }
            let mut foo = self.underlying.view();
            for (&ax, &pos) in self.renumbered_axes.iter().zip(&ret) {
                foo = foo.index_axis_move(Axis(ax), pos);
            }
            Some(foo)
        } else {
            None
        }
    }
}

use ndarray::s;

fn main() {
    let m1 = rand_array((2, 3, 4));
    println!("{:?}", einsum("ijk->i", &[&m1]));

    for p in MultiAxisIterator::new(&m1.view().into_dyn(), &[0, 2]) {
        println!("{:?}", p)
    }

    println!("");

    for i in 0..2 {
        for j in 0..4 {
            println!("{:?}", m1.slice(s![i, .., j]));
        }
    }

    // for _ in 0..1000 {
    //     let m1 = rand_array((60, 40));
    //     let m2 = rand_array((60, 40));
    //
    //     let hadamard_product = einsum("ij,ij->ij", &[&m1, &m2]).unwrap();
    //     println!("{}", hadamard_product.sum());
    // }
}
