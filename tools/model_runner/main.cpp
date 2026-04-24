// model_runner: CLI demo that builds a tiny graph (MatMul -> Add -> ReLU)
// in-memory and runs it through the IERR runtime on the chosen backend.
//
// Usage:
//   model_runner [--backend sim|npu] [--m M] [--k K] [--n N]
//
// Environment:
//   IERR_SIM_CORES=<n>   # override simulated NPU core count
//   IERR_LOG=debug|info  # logging level

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ierr/ierr.h"

namespace {

void usage() {
    std::printf(
        "Usage: model_runner [--backend sim|npu] [--m M] [--k K] [--n N]\n"
        "  Builds y = relu(a @ b + bias) and prints a checksum.\n");
}

int run_demo(ierr::Backend backend, int M, int K, int N) {
    auto rt = ierr::Runtime::Create(backend);
    if (!rt) { std::fprintf(stderr, "failed to create runtime\n"); return 2; }

    std::printf("Backend: %s, cores=%u\n", rt->caps().name, rt->caps().num_cores);

    using ierr::DType; using ierr::TensorDesc; using ierr::OpKind;
    ierr::Graph g;
    int t_a    = g.add_tensor({{M, K}, DType::F32});
    int t_b    = g.add_tensor({{K, N}, DType::F32});
    int t_mm   = g.add_tensor({{M, N}, DType::F32});
    int t_bias = g.add_tensor({{M, N}, DType::F32});
    int t_add  = g.add_tensor({{M, N}, DType::F32});
    int t_y    = g.add_tensor({{M, N}, DType::F32});

    g.ops.push_back({OpKind::MatMul, {t_a, t_b}, {t_mm}, "matmul"});
    g.ops.push_back({OpKind::Add,    {t_mm, t_bias}, {t_add}, "add_bias"});
    g.ops.push_back({OpKind::Relu,   {t_add}, {t_y}, "relu"});

    // Allocate every tensor (caller owns storage).
    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (const auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }

    // Fill inputs deterministically.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto fill = [&](int idx) {
        const auto& d = g.tensors[idx];
        std::vector<float> tmp(d.num_elements());
        for (auto& v : tmp) v = dist(rng);
        bufs[idx]->copy_from_host(tmp.data(), d.byte_size());
    };
    fill(t_a);
    fill(t_b);
    fill(t_bias);

    if (auto s = rt->Run(g, bufs); s != IERR_OK) {
        std::fprintf(stderr, "Run failed: %s\n", ierr_status_str(s));
        return 3;
    }

    std::vector<float> out(g.tensors[t_y].num_elements());
    bufs[t_y]->copy_to_host(out.data(), g.tensors[t_y].byte_size());
    double sum = 0.0; float mx = 0.f;
    for (float v : out) { sum += v; if (v > mx) mx = v; }
    std::printf("Output [%dx%d]: sum=%.6f max=%.6f first=%.6f\n",
                M, N, sum, mx, out.front());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ierr::Backend backend = ierr::Backend::Sim;
    int M = 32, K = 64, N = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int extra) { return i + extra < argc; };
        if (a == "--backend" && need(1)) {
            std::string v = argv[++i];
            if (v == "sim") backend = ierr::Backend::Sim;
            else if (v == "npu") backend = ierr::Backend::Npu;
            else { usage(); return 1; }
        } else if (a == "--m" && need(1)) { M = std::atoi(argv[++i]); }
        else if (a == "--k" && need(1)) { K = std::atoi(argv[++i]); }
        else if (a == "--n" && need(1)) { N = std::atoi(argv[++i]); }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { usage(); return 1; }
    }
    if (M <= 0 || K <= 0 || N <= 0) { usage(); return 1; }
    return run_demo(backend, M, K, N);
}
