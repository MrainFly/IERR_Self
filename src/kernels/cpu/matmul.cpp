// CPU kernel: MatMul, y[M,N] = a[M,K] * b[K,N]. Data-parallel over M (rows).
// Naive triple loop – correctness over speed in this skeleton.
#include "runtime/op_registry.h"

namespace ierr {

static int64_t dim(const TensorDesc& d, int i) {
    return (i < 0 || static_cast<size_t>(i) >= d.shape.size()) ? 0 : d.shape[i];
}

static uint32_t matmul_plan(const KernelContext& ctx, uint32_t num_cores) {
    if (!ctx.tensors || ctx.op->outputs.empty()) return 1;
    int64_t M = dim((*ctx.tensors)[ctx.op->outputs[0]]->desc(), 0);
    if (M <= 0) return 1;
    uint32_t want = static_cast<uint32_t>(M);
    if (want > num_cores) want = num_cores;
    return want;
}

static ierr_status_t matmul_run(const KernelContext& ctx) {
    const auto& op = *ctx.op;
    if (op.inputs.size() != 2 || op.outputs.size() != 1) return IERR_ERR_INVALID_ARG;
    auto* a = (*ctx.tensors)[op.inputs[0]];
    auto* b = (*ctx.tensors)[op.inputs[1]];
    auto* y = (*ctx.tensors)[op.outputs[0]];

    int64_t M = dim(a->desc(), 0);
    int64_t K = dim(a->desc(), 1);
    int64_t Kb = dim(b->desc(), 0);
    int64_t N = dim(b->desc(), 1);
    int64_t My = dim(y->desc(), 0);
    int64_t Ny = dim(y->desc(), 1);
    if (M <= 0 || K <= 0 || N <= 0 || K != Kb || M != My || N != Ny) {
        return IERR_ERR_INVALID_ARG;
    }

    const ierr_hal_t* hal = ::ierr_hal_get_sim();
    auto* pa = static_cast<const float*>(hal->mem_map(a->buffer()));
    auto* pb = static_cast<const float*>(hal->mem_map(b->buffer()));
    auto* py = static_cast<float*>(hal->mem_map(y->buffer()));

    int64_t per = (M + ctx.shard_count - 1) / ctx.shard_count;
    int64_t lo = per * static_cast<int64_t>(ctx.shard_id);
    int64_t hi = lo + per; if (hi > M) hi = M;
    for (int64_t m = lo; m < hi; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                acc += pa[m * K + k] * pb[k * N + n];
            }
            py[m * N + n] = acc;
        }
    }
    return IERR_OK;
}

void RegisterMatMulKernel() {
    OpRegistry::Instance().Register(OpKind::MatMul, Backend::Sim,
                                    KernelImpl{matmul_plan, matmul_run});
}

void RegisterCpuKernels() {
    extern void RegisterAddKernel();
    extern void RegisterReluKernel();
    RegisterAddKernel();
    RegisterReluKernel();
    RegisterMatMulKernel();
}

} // namespace ierr
