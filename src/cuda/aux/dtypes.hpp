#pragma once
// Shared scalar dtype tags for the raw-pointer launcher APIs. These replace at::ScalarType
// in the original dispatch tables; the values are arbitrary but must stay stable because
// callers pass them explicitly.

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

enum DType : int
{
    kFloat    = 0,
    kHalf     = 1,
    kBFloat16 = 2
};

}} // namespace helios::aux