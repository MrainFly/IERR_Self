// End-to-end runtime test: matmul -> add -> relu, asserting against a
// CPU reference computed in the test itself.
#include <cmath>
#include <random>
#include <vector>

#include "ierr/ierr.h"
#include "test_harness.h"

using ierr::Backend; using ierr::DType; using ierr::OpKind;

IERR_TEST(Runtime_EndToEndPipeline) {
    auto rt = ierr::Runtime::Create(Backend::Sim);
    IERR_EXPECT(rt != nullptr);

    constexpr int M = 8, K = 6, N = 4;
    ierr::Graph g;
    int ta   = g.add_tensor({{M, K}, DType::F32});
    int tb   = g.add_tensor({{K, N}, DType::F32});
    int tmm  = g.add_tensor({{M, N}, DType::F32});
    int tbias= g.add_tensor({{M, N}, DType::F32});
    int tadd = g.add_tensor({{M, N}, DType::F32});
    int ty   = g.add_tensor({{M, N}, DType::F32});
    g.ops.push_back({OpKind::MatMul, {ta, tb}, {tmm}, "mm"});
    g.ops.push_back({OpKind::Add,    {tmm, tbias}, {tadd}, "addb"});
    g.ops.push_back({OpKind::Relu,   {tadd}, {ty}, "relu"});

    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> A(M*K), B(K*N), Bias(M*N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : Bias) v = dist(rng);
    bufs[ta]->copy_from_host(A.data(), A.size()*sizeof(float));
    bufs[tb]->copy_from_host(B.data(), B.size()*sizeof(float));
    bufs[tbias]->copy_from_host(Bias.data(), Bias.size()*sizeof(float));

    IERR_EXPECT_EQ(rt->Run(g, bufs), IERR_OK);

    std::vector<float> Y(M*N);
    bufs[ty]->copy_to_host(Y.data(), Y.size()*sizeof(float));

    // Reference.
    std::vector<float> ref(M*N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m*K + k] * B[k*N + n];
            float v = s + Bias[m*N + n];
            ref[m*N + n] = v > 0 ? v : 0;
        }
    for (int i = 0; i < M*N; ++i) IERR_EXPECT_NEAR(Y[i], ref[i], 1e-4);
}

// Re-run the same graph multiple times to exercise multi-shard scheduling and
// confirm there are no races or stale-buffer issues.
IERR_TEST(Runtime_RepeatedRunsAreStable) {
    auto rt = ierr::Runtime::Create(Backend::Sim);
    IERR_EXPECT(rt != nullptr);

    ierr::Graph g;
    int ta = g.add_tensor({{1024}, DType::F32});
    int tb = g.add_tensor({{1024}, DType::F32});
    int ty = g.add_tensor({{1024}, DType::F32});
    g.ops.push_back({OpKind::Add, {ta, tb}, {ty}, "add"});

    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }
    std::vector<float> A(1024, 1.5f), B(1024, 2.25f), Y(1024);
    bufs[ta]->copy_from_host(A.data(), A.size()*sizeof(float));
    bufs[tb]->copy_from_host(B.data(), B.size()*sizeof(float));
    for (int iter = 0; iter < 10; ++iter) {
        IERR_EXPECT_EQ(rt->Run(g, bufs), IERR_OK);
        bufs[ty]->copy_to_host(Y.data(), Y.size()*sizeof(float));
        for (float v : Y) IERR_EXPECT_NEAR(v, 3.75f, 1e-6);
    }
}
