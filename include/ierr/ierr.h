// IERR Runtime – public C++ API.
//
// Minimal surface for building a graph, allocating tensors and running it on
// a chosen backend. The same calls work against the simulator (Windows/Linux
// PC) and a future real NPU backend.
#ifndef IERR_IERR_H
#define IERR_IERR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ierr/error.h"
#include "ierr/hal.h"

namespace ierr {

enum class DType : uint8_t {
    F32 = 0,
};

struct TensorDesc {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    size_t element_bytes() const;
    size_t num_elements() const;
    size_t byte_size() const;
};

// Selects which backend a Runtime targets.
enum class Backend {
    Sim, // PC simulator – default
    Npu, // real device (stub)
};

class Runtime; // fwd

// A device-resident buffer. Owns its HAL allocation.
class Tensor {
public:
    Tensor() = default;
    Tensor(Runtime* rt, TensorDesc desc);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    const TensorDesc& desc() const { return desc_; }
    ierr_hal_buffer_t buffer() const { return buf_; }

    // Convenience host<->device copies (synchronous).
    ierr_status_t copy_from_host(const void* src, size_t bytes);
    ierr_status_t copy_to_host(void* dst, size_t bytes) const;

private:
    Runtime* rt_ = nullptr;
    TensorDesc desc_{};
    ierr_hal_buffer_t buf_ = nullptr;
};

// Op kinds the bundled CPU kernel library knows about. The op registry maps
// (kind, backend) -> kernel implementation.
enum class OpKind {
    Add,    // y = a + b, elementwise
    Relu,   // y = max(0, x)
    MatMul, // y = a @ b, [M,K] x [K,N] -> [M,N]
};

struct OpNode {
    OpKind kind;
    std::vector<int> inputs;  // indices into Graph::tensors
    std::vector<int> outputs; // indices into Graph::tensors
    std::string name;
};

// A trivial linear graph IR. Enough to exercise the runtime end-to-end.
struct Graph {
    std::vector<TensorDesc> tensors;
    std::vector<OpNode>     ops;

    int add_tensor(TensorDesc d) {
        tensors.push_back(std::move(d));
        return static_cast<int>(tensors.size()) - 1;
    }
};

// Host Runtime façade. Owns the HAL device + a default stream and provides
// scheduling for a Graph. Multi-core fan-out happens inside the scheduler:
// each kernel reports how it wants to be split, the runtime issues N HAL
// commands targeting different cores.
class Runtime {
public:
    static std::unique_ptr<Runtime> Create(Backend backend = Backend::Sim);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const ierr_hal_t* hal() const { return hal_; }
    ierr_hal_device_t device() const { return dev_; }
    ierr_hal_stream_t default_stream() const { return stream_; }
    const ierr_hal_caps_t& caps() const { return caps_; }

    // Run a graph. `tensor_buffers` must be sized to graph.tensors.size() and
    // every entry must be a pre-allocated Tensor (caller owns lifetimes). Input
    // tensors should already contain their data; output tensors will be filled.
    ierr_status_t Run(const Graph& g, std::vector<Tensor*>& tensor_buffers);

private:
    Runtime() = default;
    const ierr_hal_t* hal_ = nullptr;
    ierr_hal_device_t dev_ = nullptr;
    ierr_hal_stream_t stream_ = nullptr;
    ierr_hal_caps_t caps_{};
};

} // namespace ierr

#endif // IERR_IERR_H
