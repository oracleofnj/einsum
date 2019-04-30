#![feature(test)]
extern crate test;
use test::Bencher;

use einsum;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn rand_array<Sh, D: Dimension>(shape: Sh) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    Sh: ShapeBuilder<Dim = D>,
{
    Array::random(shape, Uniform::new(-5., 5.))
}

#[bench]
fn bench_multiply_builtin_small(b: &mut Bencher) {
    let m1 = rand_array((3, 4));
    let m2 = rand_array((4, 5));

    b.iter(|| m1.dot(&m2));
}

#[bench]
fn bench_multiply_small(b: &mut Bencher) {
    let m1 = rand_array((3, 4));
    let m2 = rand_array((4, 5));

    b.iter(|| einsum::slow_einsum("ij,jk->ik", &[&m1, &m2]));
}

#[bench]
fn bench_multiply_builtin_medium(b: &mut Bencher) {
    let m1 = rand_array((300, 400));
    let m2 = rand_array((400, 500));

    b.iter(|| m1.dot(&m2));
}

#[bench]
fn bench_multiply_small_medium(b: &mut Bencher) {
    let m1 = rand_array((300, 400));
    let m2 = rand_array((400, 500));

    b.iter(|| einsum::slow_einsum("ij,jk->ik", &[&m1, &m2]));
}
