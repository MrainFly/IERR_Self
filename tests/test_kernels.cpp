// Tests for individual CPU kernels via the runtime API.
#include <vector>

#include "ierr/ierr.h"
#include "test_harness.h"

using ierr::Backend; using ierr::DType; using ierr::OpKind;

static std::unique_ptr<ierr::Runtime> mk() { return ierr::Runtime::Create(Backend::Sim); }

IERR_TEST(Kernel_Add) {
    auto rt = mk(); IERR_EXPECT(rt != nullptr);
    ierr::Graph g;
    int ta = g.add_tensor({{4}, DType::F32});
    int tb = g.add_tensor({{4}, DType::F32});
    int ty = g.add_tensor({{4}, DType::F32});
    g.ops.push_back({OpKind::Add, {ta, tb}, {ty}, "add"});

    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }
    float A[4] = {1, 2, 3, 4};
    float B[4] = {10, 20, 30, 40};
    bufs[ta]->copy_from_host(A, sizeof(A));
    bufs[tb]->copy_from_host(B, sizeof(B));

    IERR_EXPECT_EQ(rt->Run(g, bufs), IERR_OK);

    float Y[4]; bufs[ty]->copy_to_host(Y, sizeof(Y));
    IERR_EXPECT_NEAR(Y[0], 11.f, 1e-6);
    IERR_EXPECT_NEAR(Y[1], 22.f, 1e-6);
    IERR_EXPECT_NEAR(Y[2], 33.f, 1e-6);
    IERR_EXPECT_NEAR(Y[3], 44.f, 1e-6);
}

IERR_TEST(Kernel_Relu) {
    auto rt = mk(); IERR_EXPECT(rt != nullptr);
    ierr::Graph g;
    int tx = g.add_tensor({{5}, DType::F32});
    int ty = g.add_tensor({{5}, DType::F32});
    g.ops.push_back({OpKind::Relu, {tx}, {ty}, "relu"});

    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }
    float X[5] = {-2, -1, 0, 1, 2};
    bufs[tx]->copy_from_host(X, sizeof(X));
    IERR_EXPECT_EQ(rt->Run(g, bufs), IERR_OK);
    float Y[5]; bufs[ty]->copy_to_host(Y, sizeof(Y));
    IERR_EXPECT_NEAR(Y[0], 0.f, 0);
    IERR_EXPECT_NEAR(Y[1], 0.f, 0);
    IERR_EXPECT_NEAR(Y[2], 0.f, 0);
    IERR_EXPECT_NEAR(Y[3], 1.f, 0);
    IERR_EXPECT_NEAR(Y[4], 2.f, 0);
}

IERR_TEST(Kernel_MatMul) {
    auto rt = mk(); IERR_EXPECT(rt != nullptr);
    ierr::Graph g;
    // [2,3] x [3,2] = [2,2]
    int ta = g.add_tensor({{2, 3}, DType::F32});
    int tb = g.add_tensor({{3, 2}, DType::F32});
    int ty = g.add_tensor({{2, 2}, DType::F32});
    g.ops.push_back({OpKind::MatMul, {ta, tb}, {ty}, "matmul"});

    std::vector<std::unique_ptr<ierr::Tensor>> owned;
    std::vector<ierr::Tensor*> bufs;
    for (auto& d : g.tensors) {
        owned.emplace_back(std::make_unique<ierr::Tensor>(rt.get(), d));
        bufs.push_back(owned.back().get());
    }
    float A[6] = {1, 2, 3, 4, 5, 6};      // [[1,2,3],[4,5,6]]
    float B[6] = {7, 8, 9, 10, 11, 12};   // [[7,8],[9,10],[11,12]]
    bufs[ta]->copy_from_host(A, sizeof(A));
    bufs[tb]->copy_from_host(B, sizeof(B));
    IERR_EXPECT_EQ(rt->Run(g, bufs), IERR_OK);
    float Y[4]; bufs[ty]->copy_to_host(Y, sizeof(Y));
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    IERR_EXPECT_NEAR(Y[0], 58.f, 1e-4);
    IERR_EXPECT_NEAR(Y[1], 64.f, 1e-4);
    IERR_EXPECT_NEAR(Y[2], 139.f, 1e-4);
    IERR_EXPECT_NEAR(Y[3], 154.f, 1e-4);
}
