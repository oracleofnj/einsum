// Copyright 2019 Jared Samet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use ndarray::prelude::*;
use ndarray::{Data, IxDyn, LinalgScalar};

mod validation;
pub use validation::{
    validate, validate_and_size, validate_and_size_from_shapes, Contraction, OutputSize,
    SizedContraction,
};

mod optimizers;
pub use optimizers::{
    generate_optimized_path, EinsumPath, FirstStep, IntermediateStep, OperandNumPair,
    OptimizationMethod,
};

mod contractors;
use contractors::{
    PairContraction, PairContractor, SingletonContraction, SingletonContractor, TensordotGeneral,
};

pub trait ArrayLike<A> {
    fn into_dyn_view(&self) -> ArrayView<A, IxDyn>;
}

impl<A, S, D> ArrayLike<A> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn into_dyn_view(&self) -> ArrayView<A, IxDyn> {
        self.view().into_dyn()
    }
}

pub fn einsum_singleton<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    tensor: &ArrayViewD<A>,
) -> ArrayD<A> {
    let csc = SingletonContraction::new(&sized_contraction);
    csc.contract_singleton(tensor)
}

fn einsum_pair<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    lhs: &ArrayViewD<A>,
    rhs: &ArrayViewD<A>,
) -> ArrayD<A> {
    let cpc = PairContraction::new(&sized_contraction);
    cpc.contract_pair(lhs, rhs)
}

pub fn einsum_path<A>(path: &EinsumPath, operands: &[&ArrayLike<A>]) -> ArrayD<A>
where
    A: LinalgScalar,
{
    let EinsumPath {
        first_step:
            FirstStep {
                ref sized_contraction,
                ref operand_nums,
            },
        ref remaining_steps,
    } = path;

    let mut result = match operand_nums {
        None => einsum_singleton(sized_contraction, &(operands[0].into_dyn_view())),
        Some(OperandNumPair { lhs, rhs }) => einsum_pair(
            sized_contraction,
            &(operands[*lhs].into_dyn_view()),
            &(operands[*rhs].into_dyn_view()),
        ),
    };
    for step in remaining_steps.iter() {
        let IntermediateStep {
            ref sized_contraction,
            ref rhs_num,
        } = step;
        result = einsum_pair(
            &sized_contraction,
            &result.view(),
            &(operands[*rhs_num].into_dyn_view()),
        );
    }
    result
}

pub fn einsum_sc<A: LinalgScalar>(
    sized_contraction: &SizedContraction,
    operands: &[&ArrayLike<A>],
) -> ArrayD<A> {
    let path = generate_optimized_path(sized_contraction, OptimizationMethod::Naive);
    einsum_path(&path, operands)
}

pub fn einsum<A: LinalgScalar>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<ArrayD<A>, &'static str> {
    let sized_contraction = validate_and_size(input_string, operands)?;
    Ok(einsum_sc(&sized_contraction, operands))
}

// API ONLY:
pub fn tensordot<A, S, S2, D, E>(
    lhs: &ArrayBase<S, D>,
    rhs: &ArrayBase<S2, E>,
    lhs_axes: &[Axis],
    rhs_axes: &[Axis],
) -> ArrayD<A>
where
    A: ndarray::LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    assert_eq!(lhs_axes.len(), rhs_axes.len());
    let lhs_axes_copy: Vec<_> = lhs_axes.iter().map(|x| x.index()).collect();
    let rhs_axes_copy: Vec<_> = rhs_axes.iter().map(|x| x.index()).collect();
    let output_order: Vec<usize> = (0..(lhs.ndim() + rhs.ndim() - 2 * (lhs_axes.len()))).collect();
    let tensordotter = TensordotGeneral::from_shapes_and_axis_numbers(
        &lhs.shape(),
        &rhs.shape(),
        &lhs_axes_copy,
        &rhs_axes_copy,
        &output_order,
    );
    tensordotter.contract_pair(&lhs.view().into_dyn(), &rhs.view().into_dyn())
}

mod wasm_bindings;
pub use wasm_bindings::{
    slow_einsum_with_flattened_operands_as_json_string_as_json,
    validate_and_size_from_shapes_as_string_as_json, validate_as_json,
};

mod slow_versions;
pub use slow_versions::{slow_einsum, slow_einsum_given_sized_contraction};
