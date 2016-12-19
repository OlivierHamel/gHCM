#pragma OPENCL EXTENSION cl_amd_printf                  : enable

// ASSUME: compiling w/ `-x clc++` to get function overloading & such

void assert_impl(bool x, constant char* expr, constant char* file, uint line);

// \todo impl?
#if __GPU__
#define assert(expr)
#else 
#define assert(expr) assert_impl( !(expr), #expr, __FILE__, __LINE__)
#endif

#ifndef local
#define kernel
#define constant
#define local
#define global
#endif


#include "util_dim.cl"



void assert_impl(bool x, constant char* expr, constant char* file, uint line) {
    if (!x) return;
    printf("ASSERT FAILED (%d) - %s - %d: %s\n", get_local_id(0), file, (int)line, expr);
    for (;;) {}
}

template<typename T>
void workgroup_scan_pwr_2(local T            slots[/*length*/]
                         ,      size_t const length /* len is power of 2 */) {
    assert(popcount(length) == 1);
    assert(get_work_dim()   == 1);
    size_t const worker_id = get_local_id(0);

    for (size_t i = 2; i <= length; i *= 2) {
        // can't do `if (x) continue;` since we need to synchronise
        bool   const deadbeat_worker = (length / i) <= worker_id;

        size_t const hi = (worker_id + 1) * i - 1;
        size_t const lo = hi - i / 2;
        T      const t0 = deadbeat_worker ? 0 : slots[lo];
        T      const t1 = deadbeat_worker ? 0 : slots[hi];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!deadbeat_worker) slots[hi] = t0 + t1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t i = 2; i < length; i *= 2) {
        // can't do `if (x) continue;` since we need to synchronise
        bool   const deadbeat_worker = (i - 1) <= worker_id;

        size_t const k  = length / i;
        size_t const lo = (worker_id + 1) * k - 1;
        size_t const hi = lo + k / 2;
        T      const t0 = deadbeat_worker ? 0 : slots[lo];
        T      const t1 = deadbeat_worker ? 0 : slots[hi];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!deadbeat_worker) slots[hi] = t0 + t1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

template<size_t K /* len is power of 2 */>
bool workgroup_and_pwr_2(local bool(&slots)[K]) {
    assert(popcount(K)    == 1);
    assert(get_work_dim() == 1);
    size_t const worker_id = get_local_id(0);

    for (size_t i = 2; i <= K; i *= 2) {
        // can't do `if (x) continue;` since we need to synchronise
        bool   const deadbeat_worker = (K / i) <= worker_id;

        size_t const hi = (worker_id + 1) * i - 1;
        size_t const lo = hi - i / 2;
        bool   const t0 = deadbeat_worker ? 0 : slots[lo];
        bool   const t1 = deadbeat_worker ? 0 : slots[hi];
        // \todo check if we can lower this to a read_mem_fence
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!deadbeat_worker) slots[hi] = t0 && t1;
        // \todo check if we can lower this to a write_mem_fence
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return slots[K - 1];
}

template<size_t K /* len is power of 2 */>
bool workgroup_or_pwr_2(local bool (&slots)[K]) {
    assert(popcount(K)    == 1);
    assert(get_work_dim() == 1);
    size_t const worker_id = get_local_id(0);

    for (size_t i = 2; i <= K; i *= 2) {
        // can't do `if (x) continue;` since we need to synchronise
        bool   const deadbeat_worker = (K / i) <= worker_id;

        size_t const hi = (worker_id + 1) * i - 1;
        size_t const lo = hi - i / 2;
        bool   const t0 = deadbeat_worker ? 0 : slots[lo];
        bool   const t1 = deadbeat_worker ? 0 : slots[hi];
        // \todo check if we can lower this to a read_mem_fence
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!deadbeat_worker) slots[hi] = t0 || t1;
        // \todo check if we can lower this to a write_mem_fence
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return slots[K - 1];
}

template<typename T>
T workgroup_min_pwr_2(local T            slots[/*length*/]
                     ,      size_t const length /* len is power of 2 */) {
    assert(popcount(K)    == 1);
    assert(get_work_dim() == 1);
    size_t const worker_id = get_local_id(0);

    for (size_t i = 2; i <= length; i *= 2) {
        // can't do `if (x) continue;` since we need to synchronise
        bool   const deadbeat_worker = (length / i) <= worker_id;

        size_t const hi = (worker_id + 1) * i - 1;
        size_t const lo = hi - i / 2;
        T      const t0 = deadbeat_worker ? 0 : slots[lo];
        T      const t1 = deadbeat_worker ? 0 : slots[hi];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!deadbeat_worker) slots[hi] = min(t0, t1);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return slots[length - 1];
}


struct Util_ForEachInfo {
    size_t worker_id; // given ID for this job
    size_t stride;    // # of elems work group takes in at once (stride <= length)
    size_t length;    // # of elems this work item has to process
};

template<typename T, typename T2>
Util_ForEachInfo util_foreach_setup(T  const   task_size
                                   ,T2 const worker_size
                                   ,T2 const worker_id) {
    size_t const ary_len      = dim_size(  task_size);
    size_t const worker_count = dim_size(worker_size);
    size_t const passes_min   = ary_len / worker_count;
    size_t const leftovers    = ary_len % worker_count;
    
    Util_ForEachInfo info;
    info.worker_id = dim_idx(worker_size, worker_id);
    info.stride    = worker_count;
    info.length    = passes_min + (info.worker_id < leftovers ? 1 : 0);
    /*if (worker_id < 2) {
    printf("foreach_setup:\n"
           "\tworker_count =%d\n"
           "\tary_len   = %d\n"
           "\tworker_id = %d\n"
           "\tpasses    = %d\n"
           "\tleftovers = %d\n"
           "\tstride    = %d\n"
           "\tlength    = %d\n"
          ,(int)worker_count
          ,(int)ary_len
          ,(int)passes_min
          ,(int)leftovers
          ,(int)info.worker_id
          ,(int)info.stride   
          ,(int)info.length   );
    }//*/

    assert(info.stride * passes_min + leftovers == ary_len);
    return info;
}

// \hack Don't have function to pointers, nor do we have lambdas in OCL 1.2
#define UTIL_FOREACH_BGN(info, elem_idx_name)                                       \
    for (size_t foreach_i = 0; foreach_i < (info).length; ++foreach_i) {            \
        size_t const elem_idx_name = (info).stride * foreach_i + (info).worker_id;  // EOM

#define UTIL_FOREACH_END() }

#define UTIL_FOREACH_WG_BGN(length, elem_idx_name)                                  \
    { Util_ForEachInfo const aryjob_wg_info                                         \
        = util_foreach_setup(length, (size_t)WORKGROUP_SIZE, get_local_id(0));      \
      UTIL_FOREACH_BGN(aryjob_wg_info, elem_idx_name)                               \
        /*if (dim_size(length) <= elem_idx_name) printf("%d of %d @ %d by %d\n", (int)elem_idx_name, (int)dim_size(length), (int)foreach_i, (int)aryjob_wg_info.worker_id);*/\
        assert(elem_idx_name < dim_size(length));                                   \
         // EOM

#define UTIL_FOREACH_WG_END() UTIL_FOREACH_END() }


template<typename T, size_t K>
struct LocalAryChecked {
    local T& operator[](size_t const i)       { assert(i < K); return data[i]; }
          T  operator[](size_t const i) const { assert(i < K); return data[i]; }

    T data[K];
};

