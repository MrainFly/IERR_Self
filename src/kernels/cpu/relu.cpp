// CPU kernel: ReLU, y = max(0, x). Data-parallel over flat element range.
#include "runtime/op_registry.h"

namespace ierr {

static uint32_t relu_plan(const KernelContext& ctx, uint32_t num_cores) {
    if (!ctx.tensors || ctx.op->outputs.empty()) return 1;
    size_t n = (*ctx.tensors)[ctx.op->outputs[0]]->desc().num_elements();
    uint32_t want = static_cast<uint32_t>((n + 16383) / 16384);
    if (want < 1) want = 1;
    if (want > num_cores) want = num_cores;
    return want;
}

static ierr_status_t relu_run(const KernelContext& ctx) {
    const auto& op = *ctx.op;
    if (op.inputs.size() != 1 || op.outputs.size() != 1) return IERR_ERR_INVALID_ARG;
    auto* x = (*ctx.tensors)[op.inputs[0]];
    auto* y = (*ctx.tensors)[op.outputs[0]];
    if (x->desc().byte_size() != y->desc().byte_size()) return IERR_ERR_INVALID_ARG;

    const ierr_hal_t* hal = ::ierr_hal_get_sim();
    auto* px = static_cast<const float*>(hal->mem_map(x->buffer()));
    auto* py = static_cast<float*>(hal->mem_map(y->buffer()));

    size_t n = y->desc().num_elements();
    size_t per = (n + ctx.shard_count - 1) / ctx.shard_count;
    size_t lo = per * ctx.shard_id;
    size_t hi = lo + per; if (hi > n) hi = n;
    for (size_t i = lo; i < hi; ++i) py[i] = px[i] > 0.f ? px[i] : 0.f;
    return IERR_OK;
}

void RegisterReluKernel() {
    OpRegistry::Instance().Register(OpKind::Relu, Backend::Sim,
                                    KernelImpl{relu_plan, relu_run});
}

} // namespace ierr
