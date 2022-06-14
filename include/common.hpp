// Definitions required by both network.hpp and linktypes.hpp

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cstdint>
#include <cassert>

//#define PROFILER
//#define DEBUG_LEVEL 7

#ifndef PERF_LOG_LEVEL
#define PERF_LOG_LEVEL 0
#endif

#if PERF_LOG_LEVEL >= 1
// enables SimpleProfiler and PerfRec statements.
#define PROFILER
#endif


#ifndef DEBUG_LEVEL
/// 0: no debug output
#define DEBUG_LEVEL 0
#endif

#if DEBUG_LEVEL >= 1
/// information that prints once per Network
#define DEBUG_NETWORK_LEVEL
#endif

#if DEBUG_LEVEL >= 7
/// information that prints up to a handful of times per Link operation
#define DEBUG_OP_LEVEL
#endif

#if DEBUG_LEVEL >= 9
/// information that prints once per edge or node on each Link operation
#define DEBUG_EDGE_LEVEL
#endif

#include "anyvector.hpp"

namespace llrt{
    /// The type of indexes to specific dimensions of a Tensor.
    /// (Flat indexes into a Tensor values array are size_t)
    using index_t = size_t;
    struct NullOptionType{
    };
    const NullOptionType NullOption;
    
}

#endif
