// Op registry: per (OpKind, Backend) lookup of a kernel implementation. The
// registry is the place where multi-core dispatch metadata lives so that the
// scheduler stays generic.
#ifndef IERR_INTERNAL_OP_REGISTRY_H
#define IERR_INTERNAL_OP_REGISTRY_H

#include <functional>
#include <vector>

#include "ierr/ierr.h"

namespace ierr {

// Context handed to a kernel for one shard of work.
struct KernelContext {
    const Graph* graph = nullptr;
    const OpNode* op = nullptr;
    const std::vector<Tensor*>* tensors = nullptr;
    uint32_t shard_id = 0;
    uint32_t shard_count = 1;
    uint32_t core_id = 0;
};

// A kernel declares (a) how to split itself across N cores and (b) how to
// execute a given shard. For the simple kernels in this skeleton we always
// split the outer dimension; more interesting kernels can override this.
struct KernelImpl {
    // Returns desired shard count given runtime caps (#cores) and the op.
    std::function<uint32_t(const KernelContext&, uint32_t num_cores)> plan;
    // Executes one shard. Must be reentrant – called on a worker thread.
    std::function<ierr_status_t(const KernelContext&)> run;
};

class OpRegistry {
public:
    static OpRegistry& Instance();

    void Register(OpKind kind, Backend backend, KernelImpl impl);
    const KernelImpl* Lookup(OpKind kind, Backend backend) const;

private:
    struct Entry { OpKind kind; Backend backend; KernelImpl impl; };
    std::vector<Entry> entries_;
};

// Implemented in src/kernels/cpu/*.cpp – called once at runtime startup.
void RegisterCpuKernels();

} // namespace ierr

#endif
