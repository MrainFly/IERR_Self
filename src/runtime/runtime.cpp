#include "ierr/ierr.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>

#include "runtime/logging.h"
#include "runtime/op_registry.h"

namespace ierr {

// ------------------------- Tensor ----------------------------------------

Tensor::Tensor(Runtime* rt, TensorDesc desc) : rt_(rt), desc_(std::move(desc)) {
    if (rt_) {
        const auto* hal = rt_->hal();
        ierr_status_t s = hal->mem_alloc(rt_->device(), desc_.byte_size(), &buf_);
        if (s != IERR_OK) {
            IERR_ERROR("Tensor alloc failed: " << ierr_status_str(s));
            buf_ = nullptr;
        }
    }
}

Tensor::~Tensor() {
    if (rt_ && buf_) rt_->hal()->mem_free(buf_);
}

Tensor::Tensor(Tensor&& o) noexcept
    : rt_(o.rt_), desc_(std::move(o.desc_)), buf_(o.buf_) {
    o.rt_ = nullptr;
    o.buf_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        if (rt_ && buf_) rt_->hal()->mem_free(buf_);
        rt_ = o.rt_;
        desc_ = std::move(o.desc_);
        buf_ = o.buf_;
        o.rt_ = nullptr;
        o.buf_ = nullptr;
    }
    return *this;
}

ierr_status_t Tensor::copy_from_host(const void* src, size_t bytes) {
    if (!rt_ || !buf_ || !src) return IERR_ERR_INVALID_ARG;
    if (bytes > rt_->hal()->mem_size(buf_)) return IERR_ERR_INVALID_ARG;
    void* p = rt_->hal()->mem_map(buf_);
    if (!p) return IERR_ERR_INTERNAL;
    std::memcpy(p, src, bytes);
    return IERR_OK;
}

ierr_status_t Tensor::copy_to_host(void* dst, size_t bytes) const {
    if (!rt_ || !buf_ || !dst) return IERR_ERR_INVALID_ARG;
    if (bytes > rt_->hal()->mem_size(buf_)) return IERR_ERR_INVALID_ARG;
    // Make sure all device work is done before we read.
    rt_->hal()->stream_synchronize(rt_->default_stream());
    const void* p = rt_->hal()->mem_map(buf_);
    if (!p) return IERR_ERR_INTERNAL;
    std::memcpy(dst, p, bytes);
    return IERR_OK;
}

// ------------------------- Runtime ---------------------------------------

namespace {
std::once_flag g_kernel_init;
} // namespace

std::unique_ptr<Runtime> Runtime::Create(Backend backend) {
    std::call_once(g_kernel_init, [] { RegisterCpuKernels(); });

    std::unique_ptr<Runtime> rt(new Runtime());
    rt->hal_ = (backend == Backend::Sim) ? ierr_hal_get_sim() : ierr_hal_get_npu();
    if (!rt->hal_ || !rt->hal_->device_open) {
        IERR_ERROR("backend has no device_open");
        return nullptr;
    }
    if (rt->hal_->device_open(&rt->dev_) != IERR_OK) {
        IERR_ERROR("device_open failed");
        return nullptr;
    }
    rt->hal_->device_query_caps(rt->dev_, &rt->caps_);
    if (rt->hal_->stream_create(rt->dev_, &rt->stream_) != IERR_OK) {
        IERR_ERROR("stream_create failed");
        rt->hal_->device_close(rt->dev_);
        return nullptr;
    }
    IERR_INFO("ierr runtime ready: backend=" << (rt->caps_.name ? rt->caps_.name : "?")
              << " cores=" << rt->caps_.num_cores);
    return rt;
}

Runtime::~Runtime() {
    if (hal_ && stream_) hal_->stream_destroy(stream_);
    if (hal_ && dev_) hal_->device_close(dev_);
}

// ------------------------- Scheduler -------------------------------------
//
// Walk the graph in declared (topological) order. For each op:
//   1. Look up the (kind, backend) kernel.
//   2. Ask it for a shard count (data-parallel fan-out hint).
//   3. Submit `shard_count` HAL commands round-robin onto cores.
//   4. Synchronize the stream so the next op sees consistent inputs.
//
// This is the simplest correct multi-core scheduler. A future revision can
// remove the per-op sync and use Events to express true dependency chains.

namespace {
struct ShardClosure {
    const KernelImpl* impl;
    KernelContext ctx;
    std::atomic<int>* error_flag;
};
} // namespace

ierr_status_t Runtime::Run(const Graph& g, std::vector<Tensor*>& tensor_buffers) {
    if (tensor_buffers.size() != g.tensors.size()) {
        return IERR_ERR_INVALID_ARG;
    }

    // All tensors must be pre-allocated by the caller. This keeps lifetime
    // ownership outside the runtime and is symmetric with how a real device
    // runtime would expose buffer handles.
    for (size_t i = 0; i < g.tensors.size(); ++i) {
        if (tensor_buffers[i] == nullptr) {
            IERR_ERROR("tensor[" << i << "] not provided");
            return IERR_ERR_INVALID_ARG;
        }
    }

    std::atomic<int> error_flag{0};
    std::vector<std::unique_ptr<ShardClosure>> live_closures;

    for (size_t op_idx = 0; op_idx < g.ops.size(); ++op_idx) {
        const auto& op = g.ops[op_idx];
        const Backend backend = (caps_.name && std::string(caps_.name) == "sim")
            ? Backend::Sim : Backend::Npu;
        const KernelImpl* impl = OpRegistry::Instance().Lookup(op.kind, backend);
        if (!impl || !impl->run) {
            IERR_ERROR("no kernel for op '" << op.name << "' (kind=" << static_cast<int>(op.kind) << ")");
            return IERR_ERR_NOT_IMPLEMENTED;
        }

        KernelContext base_ctx;
        base_ctx.graph = &g;
        base_ctx.op = &op;
        base_ctx.tensors = &tensor_buffers;

        uint32_t shards = impl->plan ? impl->plan(base_ctx, caps_.num_cores) : 1;
        if (shards == 0) shards = 1;

        IERR_DEBUG("op[" << op_idx << "] '" << op.name << "' -> " << shards << " shard(s)");

        for (uint32_t s = 0; s < shards; ++s) {
            auto cl = std::make_unique<ShardClosure>();
            cl->impl = impl;
            cl->ctx = base_ctx;
            cl->ctx.shard_id = s;
            cl->ctx.shard_count = shards;
            cl->error_flag = &error_flag;

            ierr_hal_command_t cmd;
            cmd.fn = [](void* user, uint32_t core_id) {
                auto* c = static_cast<ShardClosure*>(user);
                c->ctx.core_id = core_id;
                ierr_status_t st = c->impl->run(c->ctx);
                if (st != IERR_OK) c->error_flag->store(static_cast<int>(st));
            };
            cmd.user = cl.get();
            cmd.preferred_core = static_cast<int32_t>(s % caps_.num_cores);

            ierr_status_t sub = hal_->stream_submit(stream_, &cmd);
            if (sub != IERR_OK) return sub;
            live_closures.emplace_back(std::move(cl));
        }

        // Sync between ops so subsequent ops read finished outputs.
        ierr_status_t sy = hal_->stream_synchronize(stream_);
        if (sy != IERR_OK) return sy;
        if (error_flag.load() != 0) {
            return static_cast<ierr_status_t>(error_flag.load());
        }
    }

    return IERR_OK;
}

} // namespace ierr
