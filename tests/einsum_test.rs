use einsum::*;
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
    let lib_output = slow_einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(correct_answer.all_close(&lib_output, TOL));
}

#[test]
fn it_computes_the_trace() {
    let square_matrix = rand_array((5, 5));

    let diag: Vec<_> = (0..square_matrix.shape()[0])
        .map(|i| square_matrix[[i, i]])
        .collect();
    let correct_answer = arr1(&diag).into_dyn();

    let lib_output = slow_einsum("ii->i", &[&square_matrix]).unwrap();

    assert!(correct_answer.all_close(&lib_output, TOL));
}

#[test]
fn it_transposes_a_matrix() {
    let rect_matrix = rand_array((2, 5));

    let correct_answer = rect_matrix.t();

    let tr1 = slow_einsum("ji", &[&rect_matrix]).unwrap();
    let tr2 = slow_einsum("ij->ji", &[&rect_matrix]).unwrap();
    let tr3 = slow_einsum("ji->ij", &[&rect_matrix]).unwrap();

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

    let sc = validate_and_size("ijkl->lij", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);

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

    let sc = validate_and_size("jkiii->jik", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);

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

    let sc = validate_and_size("kjiji->ijk", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}


#[test]
fn it_collapses_a_singleton_with_a_repeat_that_gets_summed() {
    // iij->j
    let s = rand_array((2, 2, 3));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for j in 0..3 {
        let mut res = 0.;
        for i in 0..2 {
            res += s[[i, i, j]];
        }
        correct_answer[j] = res;
    }

    let sc = validate_and_size("iij->j", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);
    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_multiple_repeats_that_get_summed() {
    // iijkk->j
    let s = rand_array((2, 2, 3, 4, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for j in 0..3 {
        let mut res = 0.;
        for i in 0..2 {
            for k in 0..4 {
                res += s[[i, i, j, k, k]];
            }
        }
        correct_answer[j] = res;
    }

    let sc = validate_and_size("iijkk->j", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);
    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}


#[test]
fn it_collapses_a_singleton_with_multiple_sums() {
    // ijk->k
    let s = rand_array((2, 3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for k in 0..4 {
        let mut res = 0.;
        for i in 0..2 {
            for j in 0..3 {
                res += s[[i, j, k]];
            }
        }
        correct_answer[k] = res;
    }

    let sc = validate_and_size("ijk->k", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);
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

    let sc = validate_and_size("iijkl->lij", &[&s]).unwrap();
    let collapsed = einsum_sc(&sc, &[&s]);

    assert!(correct_answer.into_dyn().all_close(&collapsed, TOL));
}

#[test]
fn tensordot_handles_degenerate_lhs() {
    let lhs = arr0(1.);
    let rhs = rand_array((2, 3));

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(rhs.into_dyn().all_close(&dotted, TOL));
}

#[test]
fn tensordot_handles_degenerate_rhs() {
    let lhs = rand_array((2, 3));
    let rhs = arr0(1.);

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(lhs.into_dyn().all_close(&dotted, TOL));
}

#[test]
fn tensordot_handles_degenerate_both() {
    let lhs = arr0(1.);
    let rhs = arr0(1.);

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(dotted[[]] == 1.);
}

#[test]
fn deduped_handles_dot_product() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3,));

    let correct_answer = arr0(lhs.dot(&rhs));
    let sc = validate_and_size("i,i->", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_hadamard_product() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3,));

    let correct_answer = (&lhs * &rhs).into_dyn();
    let sc = validate_and_size("i,i->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vec_vec() {
    let lhs = rand_array((3,));
    let rhs = rand_array((4,));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i]] * rhs[[j]];
        }
    }

    let sc = validate_and_size("i,j->ij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_matrix_vector_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4,));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for i in 0..3 {
        let mut res = 0.;
        for j in 0..4 {
            res += lhs[[i, j]] * rhs[[j]];
        }
        correct_answer[i] = res;
    }

    let sc = validate_and_size("ij,j->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
    assert!(correct_answer.all_close(&(lhs.dot(&rhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_2() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for j in 0..4 {
        let mut res = 0.;
        for i in 0..3 {
            res += lhs[[i]] * rhs[[i, j]];
        }
        correct_answer[j] = res;
    }

    let sc = validate_and_size("i,ij->j", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
    assert!(correct_answer.all_close(&(lhs.dot(&rhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_3() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for j in 0..4 {
        let mut res = 0.;
        for i in 0..3 {
            res += lhs[[i, j]] * rhs[[i]];
        }
        correct_answer[j] = res;
    }

    let sc = validate_and_size("ij,i->j", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
    assert!(correct_answer.all_close(&(rhs.dot(&lhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_4() {
    let lhs = rand_array((4,));
    let rhs = rand_array((3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for i in 0..3 {
        let mut res = 0.;
        for j in 0..4 {
            res += lhs[[j]] * rhs[[i, j]];
        }
        correct_answer[i] = res;
    }

    let sc = validate_and_size("j,ij->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
    assert!(correct_answer.all_close(&(rhs.dot(&lhs)), TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i, j]] * rhs[[i]];
        }
    }

    let sc = validate_and_size("ij,i->ij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_2() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array2<f64> = Array::zeros((4, 3));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[j, i]] = lhs[[i, j]] * rhs[[i]];
        }
    }

    let sc = validate_and_size("ij,i->ji", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_3() {
    let lhs = rand_array((3,));
    let rhs = rand_array((4, 3));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i]] * rhs[[j, i]];
        }
    }

    let sc = validate_and_size("i,ji->ij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3, 4));

    let correct_answer = (&lhs * &rhs).sum_axis(Axis(1));

    let sc = validate_and_size("ij,ij->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_2() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4, 3));

    let correct_answer = (&lhs * &rhs.t()).sum_axis(Axis(1));

    let sc = validate_and_size("ij,ji->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_3() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4, 3));

    let correct_answer = (&lhs * &rhs.t()).sum_axis(Axis(0));

    let sc = validate_and_size("ji,ij->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_multiple_stacked_axes() {
    let lhs = rand_array((2, 3, 4, 5));
    let rhs = rand_array((3, 5, 4, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 3, 2));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    correct_answer[[k, j, i]] += lhs[[i, j, k, l]] * rhs[[j, l, k, i]];
                }
            }
        }
    }

    let sc = validate_and_size("ijkl,jlki->kji", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer = (&lhs * &rhs).into_dyn();

    let sc = validate_and_size("ij,ij->ij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product_2() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer_t = (&lhs * &rhs).into_dyn();
    let correct_answer = correct_answer_t.t();

    let sc = validate_and_size("ij,ij->ji", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product_3() {
    let lhs = rand_array((3, 2));
    let rhs = rand_array((2, 3));

    let correct_answer = (&lhs.t() * &rhs).into_dyn();

    let sc = validate_and_size("ji,ij->ij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_double_dot_product_mat_mat() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer = arr0((&lhs * &rhs).sum());
    let sc = validate_and_size("ij,ij->", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vect_mat_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4,));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[k, i, j]] = lhs[[i, j]] * rhs[[k]];
            }
        }
    }
    let sc = validate_and_size("ij,k->kij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vect_mat_2() {
    let lhs = rand_array((4,));
    let rhs = rand_array((2, 3));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[k, i, j]] = rhs[[i, j]] * lhs[[k]];
            }
        }
    }
    let sc = validate_and_size("k,ij->kij", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_mat_mat() {
    let lhs = rand_array((5, 4));
    let rhs = rand_array((2, 3));

    let mut correct_answer: Array4<f64> = Array::zeros((4, 2, 5, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    correct_answer[[k, i, l, j]] = lhs[[l, k]] * rhs[[i, j]];
                }
            }
        }
    }
    let sc = validate_and_size("lk,ij->kilj", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}


#[test]
fn deduped_handles_matrix_product_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((3, 4));

    let correct_answer = lhs.dot(&rhs);
    let sc = validate_and_size("ij,jk", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_matrix_product_2() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4, 3));

    let correct_answer = lhs.dot(&rhs.t());
    let sc = validate_and_size("ij,kj", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_outer_product_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 3, 4));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[i, j, k]] = lhs[[i, j]] * rhs[[k, i]];
            }
        }
    }

    let sc = validate_and_size("ij,ki->ijk", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn diagonals_product() {
    let lhs = rand_array((2, 2));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i, i]] * rhs[[i, i]];
    }

    let sc = validate_and_size("ii,ii->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn diagonals_product_lhs() {
    let lhs = rand_array((2, 2));
    let rhs = rand_array((2,));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i, i]] * rhs[[i]];
    }

    let sc = validate_and_size("ii,i->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn diagonals_product_rhs() {
    let lhs = rand_array((2,));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i]] * rhs[[i, i]];
    }

    let sc = validate_and_size("i,ii->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn presum_lhs() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        for j in 0..3 {
            correct_answer[[i]] += lhs[[i, j]] * rhs[[i, i]];
        }
    }

    let sc = validate_and_size("ij,ii->i", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn it_tolerates_permuted_axes() {
    let lhs = rand_array((2, 3, 4));
    let mut rhs = rand_array((5, 3, 4));
    rhs = rhs.permuted_axes([1, 2, 0]);

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    correct_answer[[i, l]] += lhs[[i, j, k]] * rhs[[j, k, l]];
                }
            }
        }
    }

    let sc = validate_and_size("ijk,jkl->il", &[&lhs, &rhs]).unwrap();
    let dotted = einsum_sc(&sc, &[&lhs, &rhs]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices() {
    let op1 = rand_array((2, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    correct_answer[[i, l]] += op1[[i, j]] * op2[[j, k]] * op3[[k, l]];
                }
            }
        }
    }

    let sc = validate_and_size("ij,jk,kl->il", &[&op1, &op2, &op3]).unwrap();
    let dotted = einsum_sc(&sc, &[&op1, &op2, &op3]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_1() {
    let op1 = rand_array((2, 3));
    let op2 = rand_array((3, 6, 6, 7, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[i, l]] +=
                                op1[[i, j]] * op2[[j, m, m, n, k]] * op3[[k, l]];
                        }
                    }
                }
            }
        }
    }

    let sc = validate_and_size("ij,jmmnk,kl->il", &[&op1, &op2, &op3]).unwrap();
    let dotted = einsum_sc(&sc, &[&op1, &op2, &op3]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_2() {
    let op1 = rand_array((2, 6, 6, 7, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 6, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[i, l]] +=
                                op1[[i, m, m, n, j]] * op2[[j, k]] * op3[[k, m, l]];
                        }
                    }
                }
            }
        }
    }

    let sc = validate_and_size("immnj,jk,kml->il", &[&op1, &op2, &op3]).unwrap();
    let dotted = einsum_sc(&sc, &[&op1, &op2, &op3]);
    assert!(correct_answer.all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_3() {
    let op1 = rand_array((2, 6, 6, 7, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((6, 2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[m, i, l]] +=
                                op1[[i, m, m, n, j]] * op2[[j, k]] * op3[[k, l]];
                        }
                    }
                }
            }
        }
    }

    let sc = validate_and_size("immnj,jk,kl->mil", &[&op1, &op2, &op3]).unwrap();
    let dotted = einsum_sc(&sc, &[&op1, &op2, &op3]);
    assert!(correct_answer.all_close(&dotted, TOL));
}
