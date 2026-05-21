#pragma once

#include <cmath>

inline double max_vector_error(const float* got, const float* expected, int N) {
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        max_error =
            std::max(max_error, static_cast<double>(std::fabs(got[i] - expected[i])));
    }
    return max_error;
}

inline bool vectors_match(const float* got, const float* expected, int N, float tol) {
    return max_vector_error(got, expected, N) <= static_cast<double>(tol);
}
