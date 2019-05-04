// TODO: Move these to tests
fn _test_parses() {
    for test_string in &vec![
        // Explicit
        "i->",
        "ij->",
        "i->i",
        "ij,ij->ij",
        "ij,ij->",
        "ij,kl->",
        "ij,jk->ik",
        "ijk,jkl,klm->im",
        "ij,jk->ki",
        "ij,ja->ai",
        "ij,ja->ia",
        "ii->i",

        // Implicit
        "ij,k",
        "i",
        "ii",
        "ijj",
        "i,j,klm,nop",
        "ij,jk",
        "ij,ja",

        // Illegal
        "->i",
        "i,",
        "->",
        "i,,,j->k",

        // Legal parse but illegal outputs
        "i,j,k,l,m->p",
        "i,j->ijj",
    ] {
        println!("Input string: {}", test_string);
        println!("{}", einsum::validate_as_json(test_string));
        println!("");
    }
}

fn main() {
    //     test_parses();
    //
    //     let r = vec![10, 3, 4];
    //     let a = Array::<u8, _>::zeros(r);
    //
    //     let p = vec![4, 5];
    //     let b = Array::<u8, _>::zeros(p);
    //
    //     println!("{:?}", einsum::validate_and_size("cij,jk->cik", &[&a, &b]));
    //     println!("{:?}", einsum::validate_and_size("ii->i", &[&b]));
    //
    //     let b = arr2(&[[4., 5.], [2., 2.]]);
    //     println!("{:?}", einsum::validate_and_size("ii->i", &[&b]));
    //
    //     println!("");
    //
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string("cij,jk->cik", "[[10,3,4],[4,5]]")
    //     );
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string("ii->i", "[[4,5]]")
    //     );
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string("ii->i", "[[2,2]]")
    //     );
    //
    //     println!("");
    //
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string_as_json("cij,jk->cik", "[[10,3,4],[4,5]]")
    //     );
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string_as_json("ii->i", "[[4,5]]")
    //     );
    //     println!(
    //         "{:?}",
    //         einsum::validate_and_size_from_shapes_as_string_as_json("ii->i", "[[2,2]]")
    //     );
    //
    // let a = arr2(&[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
    // let b = arr2(&[
    //     [0, 10, 20, 30, 40],
    //     [50, 60, 70, 80, 90],
    //     [100, 110, 120, 130, 140],
    //     [150, 160, 170, 180, 190],
    // ]);
    // println!("{:?}\n", einsum::slow_einsum("ij,jk->ik", &[&a, &b]));
    //
    // let c = arr2(&[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
    // println!("{:?}\n", einsum::slow_einsum("ii->i", &[&c]));
    //
    // println!("{:?}\n", einsum::slow_einsum("ii->", &[&c]));
    //
    // println!("{:?}\n", einsum::slow_einsum("ii", &[&c]));
    //
    // println!("{:?}\n", einsum::slow_einsum("ji", &[&c]));
    //
    // println!("{:?}\n", einsum::slow_einsum("ji->", &[&c]));
    //
    // println!(
    //     "{:?}\n",
    //     // einsum::slow_einsum_with_flattened_operands_as_string::<f64>(
    //     einsum::slow_einsum_with_flattened_operands_as_flattened_json_string(
    //         "ij",
    //         r##"
    //         [
    //             {
    //                 "contents": [0, 1, 2, 3],
    //                 "shape": [2, 2]
    //             }
    //         ]
    //         "##
    //     )
    // )
}
