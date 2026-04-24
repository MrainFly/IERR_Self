// CPU kernel: elementwise Add, y = a + b. Splits the flat element range across
// available cores (data parallelism). Behavioral parity with a future NPU
// kernel: f32 only, IEEE-754 add.
#include <cstring>

#include "runtime/op_registry.h"

namespace ierr {

static uint32_t add_plan(const KernelContext& ctx, uint32_t num_cores) {
    if (!ctx.tensors || ctx.op->outputs.empty()) return 1;
    auto* y = (*ctx.tensors)[ctx.op->outputs[0]];
    size_t n = y->desc().num_elements();
    // 1 shard per ~16k elements, capped by core count.
    uint32_t want = static_cast<uint32_t>((n + 16383) / 16384);
    if (want < 1) want = 1;
    if (want > num_cores) want = num_cores;
    return want;
}

static ierr_status_t add_run(const KernelContext& ctx) {
    const auto& op = *ctx.op;
    if (op.inputs.size() != 2 || op.outputs.size() != 1) return IERR_ERR_INVALID_ARG;
    auto* a = (*ctx.tensors)[op.inputs[0]];
    auto* b = (*ctx.tensors)[op.inputs[1]];
    auto* y = (*ctx.tensors)[op.outputs[0]];
    if (a->desc().byte_size() != y->desc().byte_size() ||
        b->desc().byte_size() != y->desc().byte_size()) {
        return IERR_ERR_INVALID_ARG;
    }
    // Map device buffers (sim => host pointers).
    const ierr_hal_t* hal = ::ierr_hal_get_sim();
    auto* pa = static_cast<const float*>(hal->mem_map(a->buffer()));
    auto* pb = static_cast<const float*>(hal->mem_map(b->buffer()));
    auto* py = static_cast<float*>(hal->mem_map(y->buffer()));

    size_t n = y->desc().num_elements();
    size_t per = (n + ctx.shard_count - 1) / ctx.shard_count;
    size_t lo = per * ctx.shard_id;
    size_t hi = lo + per; if (hi > n) hi = n;
    for (size_t i = lo; i < hi; ++i) py[i] = pa[i] + pb[i];
    return IERR_OK;
}

void RegisterAddKernel() {
    OpRegistry::Instance().Register(OpKind::Add, Backend::Sim,
                                    KernelImpl{add_plan, add_run});
}

} // namespace ierr
