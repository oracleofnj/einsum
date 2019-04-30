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
