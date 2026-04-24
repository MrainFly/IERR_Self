#include "ierr/ierr.h"

namespace ierr {

size_t TensorDesc::element_bytes() const {
    switch (dtype) {
        case DType::F32: return 4;
    }
    return 0;
}

size_t TensorDesc::num_elements() const {
    size_t n = 1;
    for (auto d : shape) {
        if (d <= 0) return 0;
        n *= static_cast<size_t>(d);
    }
    return n;
}

size_t TensorDesc::byte_size() const {
    return num_elements() * element_bytes();
}

} // namespace ierr
