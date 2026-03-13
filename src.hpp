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
    // Q has shape [i+1, d], K[j] has shape [1, d], V[j] has shape [1, d]

    // Move current query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Accumulator for summing all attention outputs
    Matrix* sum_result = nullptr;

    // Process each key-value pair
    for (size_t j = 0; j <= i; ++j) {
      // Move K[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[j]);

      // Transpose K[j] for multiplication: K^T has shape [d, 1]
      gpu_sim.Transpose(keys[j]);

      // Compute Q * K^T: [i+1, d] * [d, 1] = [i+1, 1]
      Matrix* qk = matrix_memory_allocator.Allocate("qk_" + std::to_string(j));
      gpu_sim.MatMul(current_query, keys[j], qk);

      // Apply softmax row-wise
      // For each row in qk, we need: softmax(row) = exp(row) / sum(exp(row))
      Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk_" + std::to_string(j));
      gpu_sim.MatExp(qk, exp_qk);

      // Sum each row to get normalization factors
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(j));
      gpu_sim.Sum(exp_qk, sum_exp);

      // Divide by sum to complete softmax
      Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_" + std::to_string(j));
      gpu_sim.MatDiv(exp_qk, sum_exp, softmax_result);

      // Move V[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(values[j]);

      // Multiply softmax result with V[j]: [i+1, 1] * [1, d] = [i+1, d]
      Matrix* attention_out = matrix_memory_allocator.Allocate("attn_out_" + std::to_string(j));
      gpu_sim.MatMul(softmax_result, values[j], attention_out);

      // Accumulate results
      if (sum_result == nullptr) {
        sum_result = attention_out;
      } else {
        Matrix* new_sum = matrix_memory_allocator.Allocate("sum_" + std::to_string(j));
        gpu_sim.MatAdd(sum_result, attention_out, new_sum);
        sum_result = new_sum;
      }

      // Clean up intermediate matrices
      gpu_sim.ReleaseMatrix(qk);
      gpu_sim.ReleaseMatrix(exp_qk);
      gpu_sim.ReleaseMatrix(sum_exp);
      gpu_sim.ReleaseMatrix(softmax_result);
      if (attention_out != sum_result) {
        gpu_sim.ReleaseMatrix(attention_out);
      }
    }

    // Move final result to HBM
    gpu_sim.MoveMatrixToGpuHbm(sum_result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(sum_result);
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