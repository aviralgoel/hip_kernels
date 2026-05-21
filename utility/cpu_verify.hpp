#pragma once

#include <cmath>

inline void cpu_rms_norm(const float* input, float gamma, float beta, float eps, float* output,
                         int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_sq += static_cast<double>(input[i]) * static_cast<double>(input[i]);
    }
    const float rms = std::sqrt(static_cast<float>(sum_sq / N) + eps);
    for (int i = 0; i < N; ++i) {
        output[i] = gamma * (input[i] / rms) + beta;
    }
}
