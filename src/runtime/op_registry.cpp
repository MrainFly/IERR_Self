#include "runtime/op_registry.h"

#include <mutex>

namespace ierr {

OpRegistry& OpRegistry::Instance() {
    static OpRegistry r;
    return r;
}

void OpRegistry::Register(OpKind kind, Backend backend, KernelImpl impl) {
    entries_.push_back({kind, backend, std::move(impl)});
}

const KernelImpl* OpRegistry::Lookup(OpKind kind, Backend backend) const {
    for (const auto& e : entries_) {
        if (e.kind == kind && e.backend == backend) return &e.impl;
    }
    return nullptr;
}

} // namespace ierr
