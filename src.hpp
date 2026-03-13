#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Keep concatenated keys and values across rounds to avoid rebuilding
  Matrix* all_keys = nullptr;
  Matrix* all_values = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build concatenated keys incrementally
    if (i == 0) {
      // First round: just copy the first key
      gpu_sim.MoveMatrixToSharedMem(keys[0]);
      all_keys = matrix_memory_allocator.Allocate("all_keys");
      gpu_sim.Copy(keys[0], all_keys, Position::kInSharedMemory);
    } else {
      // Subsequent rounds: concatenate new key to existing
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
      Matrix* new_keys = matrix_memory_allocator.Allocate("all_keys_new");
      gpu_sim.Concat(all_keys, keys[i], new_keys, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(all_keys);
      all_keys = new_keys;
    }

    // Build concatenated values incrementally
    if (i == 0) {
      gpu_sim.MoveMatrixToSharedMem(values[0]);
      all_values = matrix_memory_allocator.Allocate("all_values");
      gpu_sim.Copy(values[0], all_values, Position::kInSharedMemory);
    } else {
      gpu_sim.MoveMatrixToSharedMem(values[i]);
      Matrix* new_values = matrix_memory_allocator.Allocate("all_values_new");
      gpu_sim.Concat(all_values, values[i], new_values, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(all_values);
      all_values = new_values;
    }

    // Make a copy of all_keys before transpose (since transpose is in-place)
    Matrix* keys_copy = matrix_memory_allocator.Allocate("keys_copy");
    gpu_sim.Copy(all_keys, keys_copy, Position::kInSharedMemory);
    gpu_sim.Transpose(keys_copy, Position::kInSharedMemory);

    // Compute Q * Keys^T: [i+1, d] * [d, i+1] = [i+1, i+1]
    Matrix* qk = matrix_memory_allocator.Allocate("qk");
    gpu_sim.MatMul(current_query, keys_copy, qk);
    gpu_sim.ReleaseMatrix(keys_copy);

    // Apply softmax row-wise
    Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
    gpu_sim.MatExp(qk, exp_qk);
    gpu_sim.ReleaseMatrix(qk);

    // Compute softmax row by row
    Matrix* softmax_result = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.GetRow(exp_qk, row, exp_row, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(exp_row, row_sum);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (softmax_result == nullptr) {
        softmax_result = softmax_row;
      } else {
        Matrix* new_softmax = matrix_memory_allocator.Allocate("softmax_new");
        gpu_sim.Concat(softmax_result, softmax_row, new_softmax, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_result = new_softmax;
      }
    }
    gpu_sim.ReleaseMatrix(exp_qk);

    // Make a copy of all_values for multiplication
    Matrix* values_copy = matrix_memory_allocator.Allocate("values_copy");
    gpu_sim.Copy(all_values, values_copy, Position::kInSharedMemory);

    // Final multiplication
    Matrix* attention_out = matrix_memory_allocator.Allocate("attention_out");
    gpu_sim.MatMul(softmax_result, values_copy, attention_out);
    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(values_copy);

    // Move to HBM for submission
    gpu_sim.MoveMatrixToGpuHbm(attention_out);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attention_out);
  }

  // Clean up persistent matrices (if needed)
  if (all_keys) gpu_sim.ReleaseMatrix(all_keys);
  if (all_values) gpu_sim.ReleaseMatrix(all_values);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu