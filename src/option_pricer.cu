#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "enums.hpp"

__global__ void monteCarloKernel(
    float *results,
    int num_paths,
    float S, float K, float r, float sigma, float T,
    float barrier,
    int steps,
    OptionType option_type,
    BarrierType barrier_type,
    OptionStyle style,
    unsigned long long seed
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float dt = T / steps;
    float st = S;
    float sum_for_average = 0.0f;
    bool knocked_out = false;

    for (int i = 0; i < steps; ++i) {
        float z = curand_normal(&state);
        st *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);

        if (option_type == OptionType::ASIAN) {
            sum_for_average += st;
        }

        if (option_type == OptionType::BARRIER) {
            if (barrier_type == BarrierType::UP_AND_OUT && st >= barrier) {
                knocked_out = true;
            }
            if (barrier_type == BarrierType::DOWN_AND_OUT && st <= barrier) {
                knocked_out = true;
            }
        }
    }

    float payoff = 0.0f;
    if (option_type == OptionType::EUROPEAN) {
        payoff = (style == OptionStyle::CALL) ? fmaxf(st - K, 0.0f) : fmaxf(K - st, 0.0f);
    }
    else if (option_type == OptionType::ASIAN) {
        float avg = sum_for_average / steps;
        payoff = (style == OptionStyle::CALL) ? fmaxf(avg - K, 0.0f) : fmaxf(K - avg, 0.0f);
    }
    else if (option_type == OptionType::BARRIER && !knocked_out) {
        payoff = (style == OptionStyle::CALL) ? fmaxf(st - K, 0.0f) : fmaxf(K - st, 0.0f);
    }

    results[idx] = expf(-r * T) * payoff;
}

void monteCarloKernelLauncher(
    at::Tensor results,
    int num_paths,
    float S, float K, float r, float sigma, float T,
    float barrier,
    int steps,
    OptionType option_type,
    BarrierType barrier_type,
    OptionStyle style,
    unsigned long long seed
)
{
    const int blockSize = 256;
    const int gridSize = (num_paths + blockSize - 1) / blockSize;

    monteCarloKernel<<<gridSize, blockSize>>>(
        results.data_ptr<float>(), num_paths,
        S, K, r, sigma, T, barrier, steps,
        option_type, barrier_type, style, seed
    );
}
