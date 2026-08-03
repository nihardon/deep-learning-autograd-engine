#include "ops.h"
#include <stdexcept>
#include <cmath>

// Detect 
#if defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    #define PLATFORM_NAME "Apple/ARM NEON"
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define PLATFORM_NAME "Intel AVX2"
#else
    #define PLATFORM_NAME "Generic CPU"
#endif
namespace ops {
    void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C) {
        int M = A.get_shape()[0];
        int K = A.get_shape()[1];
        int N = B.get_shape()[1];
        
        // Quick error check
        if (B.get_shape()[0] != K) throw std::runtime_error("Dimension mismatch");

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
    }

    void matmul(const Tensor& A, const Tensor& B, Tensor& C){

        int M = A.get_shape()[0];
        int K = A.get_shape()[1];
        int N = B.get_shape()[1];
        C.fill(0.0f);

    #if defined(__aarch64__) || defined(__ARM_NEON)
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {

                // "vdupq_n_f32" = Vector Duplicate Quad (4) Float 32
                float32x4_t a_vec = vdupq_n_f32(A(i, k));

                int j = 0;

                // Loop stride is 4 because NEON holds 4 floats
                for (; j <= N - 4; j += 4) {

                    // "vld1q_f32" = Vector Load 1 Quad Float 32
                    float32x4_t b_vec = vld1q_f32(&B(k, j));
                    float32x4_t c_vec = vld1q_f32(&C(i, j));

                    // "vfmaq_f32" = Vector Fused Multiply Add Quad (C + A * B)
                    float32x4_t result = vfmaq_f32(c_vec, a_vec, b_vec);

                    vst1q_f32(&C(i, j), result);
                }

                // Cleanup loop for remaining columns
                for (; j < N; j++) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }

    #elif defined(__AVX2__)
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                __m256 a_vec = _mm256_set1_ps(A(i, k));

                int j = 0;
                for (; j <= N - 8; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&B(k, j));
                    __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                    __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(&C(i, j), result);
                }
                
                for (; j < N; j++) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }

    #else
        // Fallback if no SIMD available
        matmul_naive(A, B, C);
    #endif
    }

    void relu(Tensor& Z) {
        float* data = Z.data();
        int size = Z.get_size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            if (data[i] < 0) {
                data[i] = 0.0f;
            }
        }
    }

    void relu_backward(const Tensor& Grad_Out, const Tensor& Input, Tensor& Grad_In){
        const float* grad_out_ptr = Grad_Out.data();
        const float* input_ptr = Input.data();
        float* grad_in_ptr = Grad_In.data();
        int size = Input.get_size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            // Derivative is 1 if input > 0, else 0
            float mask = (input_ptr[i] > 0.0f) ? 1.0f : 0.0f;
            
            grad_in_ptr[i] += grad_out_ptr[i] * mask;
        }
    }

    void transpose(const Tensor& A, Tensor& B) {
        int rows = A.get_shape()[0];
        int cols = A.get_shape()[1];

        // B must be [cols, rows]
        if (B.get_shape()[0] != cols || B.get_shape()[1] != rows) {
            throw std::runtime_error("Transpose shape mismatch");
        }

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // A(i, j) -> B(j, i)
                B(j, i) = A(i, j);
            }
        }
    }

    void softmax(Tensor& Z) {
        int rows = Z.get_shape()[0];
        int cols = Z.get_shape()[1];

        // Softmax is applied row wise 
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            float max_val = -1e9;
            for (int j = 0; j < cols; j++) {
                if (Z(i, j) > max_val) max_val = Z(i, j);
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                float e = std::exp(Z(i, j) - max_val);
                Z(i, j) = e;
                sum += e;
            }

            for (int j = 0; j < cols; j++) {
                Z(i, j) /= sum;
            }
        }
    }

    void div_scalar(Tensor& A, float scalar) {
        float* data = A.data();
        int size = A.get_size();

        if (scalar == 0.0f) {
            throw std::runtime_error("Division by zero");
        }

        // Multiply by reciprocal 
        float inv_scalar = 1.0f / scalar;

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            data[i] *= inv_scalar;
        }
    }

    void add(const Tensor& A, const Tensor& B, Tensor& C) {
        if (A.get_size() != B.get_size() || A.get_size() != C.get_size()) {
            throw std::runtime_error("Shape mismatch in add");
        }

        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();
        int size = A.get_size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }

    void sgd_step(Tensor& param, const Tensor& grad, float learning_rate) {
        float* p_ptr = param.data();
        const float* g_ptr = grad.data();
        int size = param.get_size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            p_ptr[i] -= learning_rate * g_ptr[i];
        }
    }
}
