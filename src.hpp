#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // For round i, we need to compute attention with keys[0..i] and values[0..i]
    // Q has shape [i+1, d], each K[j] has shape [1, d], each V[j] has shape [1, d]
    // We need to concatenate all keys and values first

    // Move current query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Concatenate all keys K[0]...K[i] to form Keys matrix [i+1, d]
    Matrix* all_keys = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      if (all_keys == nullptr) {
        // First key, just copy it
        all_keys = matrix_memory_allocator.Allocate("all_keys");
        gpu_sim.Copy(keys[j], all_keys, Position::kInSharedMemory);
      } else {
        // Concatenate with existing keys (axis=0 for vertical stacking)
        Matrix* new_keys = matrix_memory_allocator.Allocate("all_keys_" + std::to_string(j));
        gpu_sim.Concat(all_keys, keys[j], new_keys, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(all_keys);
        all_keys = new_keys;
      }
    }

    // Concatenate all values V[0]...V[i] to form Values matrix [i+1, d]
    Matrix* all_values = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      if (all_values == nullptr) {
        all_values = matrix_memory_allocator.Allocate("all_values");
        gpu_sim.Copy(values[j], all_values, Position::kInSharedMemory);
      } else {
        Matrix* new_values = matrix_memory_allocator.Allocate("all_values_" + std::to_string(j));
        gpu_sim.Concat(all_values, values[j], new_values, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(all_values);
        all_values = new_values;
      }
    }

    // Transpose Keys for multiplication: Keys^T has shape [d, i+1]
    gpu_sim.Transpose(all_keys, Position::kInSharedMemory);

    // Compute Q * Keys^T: [i+1, d] * [d, i+1] = [i+1, i+1]
    Matrix* qk = matrix_memory_allocator.Allocate("qk");
    gpu_sim.MatMul(current_query, all_keys, qk);

    // Apply softmax row-wise on [i+1, i+1] matrix
    // For softmax, we need: softmax(row) = exp(row) / sum(exp(row))
    Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
    gpu_sim.MatExp(qk, exp_qk);

    // For each row, compute its sum and then divide
    // Since Sum returns a 1x1 matrix, we need to sum each row separately
    // This is tricky... let me think about how to do row-wise operations

    // Actually, looking at the problem description, MatDiv can work with a row vector
    // Let me check if there's a row-wise sum operation...
    // For now, let's assume we need to compute softmax for each row

    // Actually, for proper softmax, I need to:
    // 1. For each row i: compute sum of exp(row[i])
    // 2. Divide each element in row[i] by that sum

    // This requires getting each row, summing it, and dividing
    Matrix* softmax_result = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(row));
      gpu_sim.GetRow(exp_qk, row, exp_row, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(exp_row, row_sum);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);

      if (softmax_result == nullptr) {
        softmax_result = softmax_row;
      } else {
        Matrix* new_softmax = matrix_memory_allocator.Allocate("softmax_concat_" + std::to_string(row));
        gpu_sim.Concat(softmax_result, softmax_row, new_softmax, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        softmax_result = new_softmax;
      }

      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);
      if (softmax_row != softmax_result) {
        gpu_sim.ReleaseMatrix(softmax_row);
      }
    }

    // Multiply softmax result with Values: [i+1, i+1] * [i+1, d] = [i+1, d]
    Matrix* attention_out = matrix_memory_allocator.Allocate("attention_out");
    gpu_sim.MatMul(softmax_result, all_values, attention_out);

    // Move final result to HBM
    gpu_sim.MoveMatrixToGpuHbm(attention_out);

    // Clean up
    gpu_sim.ReleaseMatrix(all_keys);
    gpu_sim.ReleaseMatrix(all_values);
    gpu_sim.ReleaseMatrix(qk);
    gpu_sim.ReleaseMatrix(exp_qk);
    gpu_sim.ReleaseMatrix(softmax_result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attention_out);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu