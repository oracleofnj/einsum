# Einsum (Einstein Summation) for [Rust ndarray](https://docs.rs/ndarray/)

General algorithm description in semi-Rust pseudocode

```
FirstStep = Singleton({
  contraction: Contraction,
}) | Pair({
  contraction: Contraction,
  lhs: usize,
  rhs: usize
})

IntermediateStep = {
  contraction: Contraction,
  rhs: usize
}

ContractionOrder = {
  first_step: FirstStep,
  remaining_steps: Vec<IntermediateStep>,
}

path: ContractionOrder = Optimize(&Contraction, &[OperandShapes]);

result: ArrayD<A> = einsum_path<A>(Path, &[&ArrayLike<A>]);

einsum_path() {
  let mut result = match first_step {
    Singleton => einsum_singleton(contraction, operands[0]),
    Pair => einsum_pair(contraction, operands[lhs], operands[rhs])
  }
  for step in remaining_steps.iter() {
    result = einsum_pair(contraction, &result, operands[rhs])
  }
  result
}

einsum_singleton() {
  // Diagonalizes repeated indices and then sums across indices that don't appear in the output
}

einsum_pair() {
  // First uses einsum_singleton to reduce lhs and rhs to tensors with no repeated indices and where
  // each index is either in the other tensor or in the output
  //
  // If there are any "stack" indices that appear in both tensors and the output, these are not
  // contracted and just used for identifying common elements. These get moved to the front of
  // the tensor and temporarily reshaped into a single dimension. Then einsum_pair_base does
  // the contraction for each subview along that dimension.
}

einsum_pair_base() {
  // Figures out the indices for LHS and RHS that are getting contracted
  // Calls tensordot on the two tensors
  // Permutes the result into the desired output order
}

tensordot() {
  // Permutes LHS so the contracted indices are at the end and permutes RHS so the contracted
  // indices are at the front. Then calls tensordot_fixed_order with the number of contracted indices
}

tensordot_fixed_order() {
  // Reshapes (previously-permuted) LHS and (previously-permuted) RHS into 2-D matrices
  // where, for LHS, the number of rows is the product of the uncontracted dimensions and the number of
  // columns is the product of the contracted dimensions, and vice-versa for RHS. Result is an MxN matrix
  // where M is the dimensionality of uncontracted LHS and N is dimensionality of uncontracted RHS.
  // Finally is reshaped back into (...uncontracted LHS shape, ...uncontracted RHS shape).
}
```
