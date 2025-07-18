#include <torch/extension.h>
#include "enums.hpp"

// Forward declaration
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
);

// Python-callable wrapper
void runMonteCarlo(
    at::Tensor results,
    int num_paths,
    float S, float K, float r, float sigma, float T,
    float barrier,
    int steps,
    OptionType option_type,
    BarrierType barrier_type,
    OptionStyle style,
    unsigned long long seed
) {
    monteCarloKernelLauncher(results, num_paths, S, K, r, sigma, T, barrier, steps,
                             option_type, barrier_type, style, seed);
}

// Pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::enum_<OptionType>(m, "OptionType")
        .value("EUROPEAN", OptionType::EUROPEAN)
        .value("ASIAN",     OptionType::ASIAN)
        .value("BARRIER",   OptionType::BARRIER)
        .export_values();

    pybind11::enum_<BarrierType>(m, "BarrierType")
        .value("NONE",         BarrierType::NONE)
        .value("UP_AND_OUT",   BarrierType::UP_AND_OUT)
        .value("DOWN_AND_OUT", BarrierType::DOWN_AND_OUT)
        .export_values();

    pybind11::enum_<OptionStyle>(m, "OptionStyle")
        .value("CALL", OptionStyle::CALL)
        .value("PUT",  OptionStyle::PUT)
        .export_values();

    m.def("runMonteCarlo", &runMonteCarlo, "GPU Monte Carlo Option Pricer");
}
