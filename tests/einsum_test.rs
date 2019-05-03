#![feature(test)]
use einsum;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
const TOL: f64 = 1e-10;

fn rand_array<Sh, D: Dimension>(shape: Sh) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    Sh: ShapeBuilder<Dim = D>,
{
    Array::random(shape, Uniform::new(-5., 5.))
}

#[test]
fn it_multiplies_two_matrices() {
    let a = rand_array((3, 4));
    let b = rand_array((4, 5));

    let correct_answer = a.dot(&b).into_dyn();
    let lib_output = einsum::slow_einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(correct_answer.all_close(&lib_output, TOL));
}

#[test]
fn it_computes_the_trace() {
    let square_matrix = rand_array((5, 5));

    let diag: Vec<_> = (0..square_matrix.shape()[0])
        .map(|i| square_matrix[[i, i]])
        .collect();
    let correct_answer = arr1(&diag).into_dyn();

    let lib_output = einsum::slow_einsum("ii->i", &[&square_matrix]).unwrap();

    assert!(correct_answer.all_close(&lib_output, TOL));
}

#[test]
fn it_transposes_a_matrix() {
    let rect_matrix = rand_array((2, 5));

    let correct_answer = rect_matrix.t();

    let tr1 = einsum::slow_einsum("ji", &[&rect_matrix]).unwrap();
    let tr2 = einsum::slow_einsum("ij->ji", &[&rect_matrix]).unwrap();
    let tr3 = einsum::slow_einsum("ji->ij", &[&rect_matrix]).unwrap();

    assert!(correct_answer.all_close(&tr1, TOL));
    assert!(correct_answer.all_close(&tr2, TOL));
    assert!(correct_answer.all_close(&tr3, TOL));
}

#[test]
fn it_collapses_a_singleton_without_repeats() {
    // ijkl->lij
    let s = rand_array((4, 2, 3, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((5, 4, 2));
    for l in 0..5 {
        for i in 0..4 {
            for j in 0..2 {
                let mut r = 0.;
                for k in 0..3 {
                    r += s[[i, j, k, l]];
                }
                correct_answer[[l, i, j]] = r;
            }
        }
    }

    let sc = einsum::validate_and_size("ijkl->lij", &[&s]).unwrap();
    let collapsed = einsum::einsum_singleton(&sc, &s);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}

#[test]
fn it_diagonalizes_a_singleton() {
    // jkiii
    let s = rand_array((2, 3, 4, 4, 4));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 4, 3));
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                correct_answer[[j, i, k]] = s[[j, k, i, i, i]];
            }
        }
    }

    let collapsed = einsum::diagonalize_singleton(&s, &[2, 3, 4], 1);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_multiple_repeats() {
    // kjiji->ijk
    let s = rand_array((4, 3, 2, 3, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 3, 4));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[i, j, k]] = s[[k, j, i, j, i]];
            }
        }
    }

    let sc = einsum::validate_and_size("kjiji->ijk", &[&s]).unwrap();
    let collapsed = einsum::einsum_singleton(&sc, &s);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}


#[test]
fn it_collapses_a_singleton() {
    // iijkl->lij
    let s = rand_array((4, 4, 2, 3, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((5, 4, 2));
    for l in 0..5 {
        for i in 0..4 {
            for j in 0..2 {
                let mut r = 0.;
                for k in 0..3 {
                    r += s[[i, i, j, k, l]];
                }
                correct_answer[[l, i, j]] = r;
            }
        }
    }

    let sc = einsum::validate_and_size("iijkl->lij", &[&s]).unwrap();
    let collapsed = einsum::einsum_singleton(&sc, &s);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}
