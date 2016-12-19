
#pragma OPENCL EXTENSION cl_amd_printf                  : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store  : enable


// ASSUME: compiling w/ `-x clc++` to get function overloading & such

void assert_impl(bool x, constant char* expr, constant char* file, uint line);

// \todo impl?
#ifdef DEVICE_IS_CPU
    #define assert(expr) assert_impl( !(expr), #expr, __FILE__, __LINE__)
#else 
    #define assert(expr)
#endif


#ifndef ENABLE_CACHE_TIME
#define ENABLE_CACHE_TIME 0
#endif

#ifndef ENABLE_CACHE_COST
#define ENABLE_CACHE_COST 0
#endif

#ifndef BLOCK_SIZE_MAX
    #define BLOCK_SIZE_MAX 32
#endif

#if 256 < BLOCK_SIZE_MAX
    #error BLOCK_SIZE_MAX must be less than 256 for the open list definition to work.
#endif

// INVARIANT: BLOCK_SIZE_MAX == get_group_size(0) && BLOCK_SIZE_MAX == get_group_size(1)
#ifndef WORKGROUP_SIZE
    #define WORKGROUP_SIZE 64
#endif

#if (BLOCK_SIZE_MAX * BLOCK_SIZE_MAX) < WORKGROUP_SIZE
    #error "Workers will be hanging around idle..."
#endif

// +1 on each border b/c we need room for neighbouring cell
#define BLOCK_PADDING                1
#define BLOCK_SIZE_MAX_PADDED        (BLOCK_SIZE_MAX + (BLOCK_PADDING * 2)          )
#define BLOCK_SIZE_MAX_PADDED_SQUARE (BLOCK_SIZE_MAX_PADDED * BLOCK_SIZE_MAX_PADDED )

#ifndef local
#define kernel
#define constant
#define local
#define global
#endif

struct BlockStatus {
    uint  solves;
    uint  cells_updated;
    float smallest_time;
    uint  border[4];
};

struct BlockJob {
    uint  id;
    uint  border;
};


constant char         k_hello_world_string[] = "Hello World\n";

constant sampler_t  k_sampler_direct_clamp
    = CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;



kernel void hello(global char* out) {
    size_t tid  = get_global_id(0);
    out[tid]    = k_hello_world_string[tid];
}

template<typename T, size_t K>
struct LocalAryChecked {
    local T& operator[](size_t const i)       { assert(i < K); return data[i]; }
          T  operator[](size_t const i) const { assert(i < K); return data[i]; }

    T data[K];
};

template<typename T, size_t K>
size_t length_of(T(&)[K]) { return K; }

template<typename T, size_t K>
inline void sort_bubble(T (&ary)[K]) {
    for (uint j = 0; (j + 1) < K; ++j) {
        for (uint i = 0; (i + 1) < K; ++i) {
            T const a = ary[i     ];
            T const b = ary[i + 1 ];
            ary[i     ] = min(a, b);
            ary[i + 1 ] = max(a, b);
        }
    }
}

template<>
void sort_bubble<float, 2>(float (&ary)[2]) {
    float const a = ary[0];
    float const b = ary[1];
    ary[0] = min(a, b);
    ary[1] = max(a, b);
}

 int2   to_i2 ( char2 v) { return ( int2 )(v.x, v.y); }
 int2   to_i2 (uchar2 v) { return ( int2 )(v.x, v.y); }
 int2   to_i2 (uint2  v) { assert(all(v < INT_MAX));
                           return ( int2 )(v.x, v.y); }
uint2   to_ui2(uchar2 v) { return (uint2 )(v.x, v.y); }
uchar2  to_uc2(uint2  v) { assert(all(v < 256));
                           return (uchar2)(v.x, v.y); }
uchar2  to_uc2( int2  v) { assert(all(0 <= v)); assert(all(v < 256));
                           return (uchar2)(v.x, v.y); }

size_t  dim_size(size_t sz) { return sz; }
size_t  dim_size(uchar2 sz) { return sz.x * sz.y; }
size_t  dim_size(uint2  sz) { return sz.x * sz.y; }
size_t  dim_size(uint3  sz) { return sz.x * sz.y * sz.z; }


bool    dim_in_bounds(uint2 sz, uint2 i) { return all(i < sz); }
bool    dim_in_bounds( int2 sz,  int2 i) {
    assert(all(0 <= sz));
    return all((0 <= i) && (i < sz));
}

size_t  dim_idx (size_t sz, size_t i) { return i                                   ; }
size_t  dim_idx (uint2  sz, uint2  i) { assert(all(i < sz));
                                        return i.x + i.y * sz.x                    ; }
size_t  dim_idx (uchar2 sz, uchar2 i) { assert(all(i < sz));
                                        return dim_idx(to_ui2(sz), to_ui2(i))      ; }
size_t  dim_idx (uint3  sz, uint3  i) { assert(all(i < sz)); 
                                        return i.x + i.y * sz.x + i.z * sz.x * sz.y; }

uint2   dim_pos(uint2  sz, size_t idx) { assert(idx < dim_size(sz));
                                         return (uint2 )(idx % sz.x, idx / sz.x); }
uchar2  dim_pos(uchar2 sz, ushort idx) { assert(idx < dim_size(sz)); 
                                         return (uchar2)(idx % sz.x, idx / sz.x); }

// unpadded size & coord -> padded idx
size_t dim_idx_padded(uint padding, uint2 size, uint2 pos)
{ return dim_idx(padding * 2 + size, padding + pos); }

size_t dim_idx_padded(uint padding, uint2 size,  int2 pos) {
    int const padding_i = (int)padding;
    assert(all(-padding_i <=       pos              ));
    //if (!all(pos       <  to_i2(size + padding_i)))
    //    printf("%d, %d  of  %d, %d", pos.x, pos.y, to_i2(size + padding_i).x, to_i2(size + padding_i).y);
    assert(all( pos       <  to_i2(size + padding_i)));
    return dim_idx(padding * 2 + size, as_uint2(padding_i + pos));
}

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





typedef LocalAryChecked<float, BLOCK_SIZE_MAX * BLOCK_SIZE_MAX> BlockCache;

struct BlockInfo {
    volatile uint solves;
    volatile uint cells_updated;

    uint2  field_size;
    uint   id;
    uint2  xy;
    uint2  num;
    uint2  origin;
    uchar2 size;
    float  smallest_time[WORKGROUP_SIZE];
#if ENABLE_CACHE_TIME
    BlockCache cache_time;
#endif
#if ENABLE_CACHE_COST
    BlockCache cache_cost;
#endif

    bool in_bounds       (int2 const local_pos) const
    { return dim_in_bounds(to_i2(size), local_pos); }
    bool in_bounds_padded(int2 const local_pos) const
    { return dim_in_bounds(to_i2(size) + int2(BLOCK_PADDING * 2)
                          ,local_pos   + int2(BLOCK_PADDING    )); }
    bool in_bounds_field (int2 const field_pos) const
    { return dim_in_bounds(to_i2(field_size), field_pos); }

    size_t field_idx(int2 const local_pos) const {
        int2 const field_pos   = to_i2(origin) + local_pos;
        assert(in_bounds_field(field_pos));
        return dim_idx(field_size, as_uint2(field_pos));
    }

    template<typename T>
    T field_get(T const default_value, global T const field[], uchar2 const local_pos) const
    { return field_get(default_value, field, to_i2(local_pos)); }

    template<typename T>
    T field_get(T const default_value, global T const field[], int2 const local_pos) const {
        //printf("fetch %d, %d (default %f)\n", local_pos.x, local_pos.y, default_value);
        assert(in_bounds_padded(local_pos));

        int2 const field_pos = to_i2(origin) + local_pos;
        if (!in_bounds_field(field_pos)) return default_value;

        return field[field_idx(local_pos)];
    }

    template<typename T>
    void field_set(global T field[], uchar2 const local_pos, T const value) const
    { return field_set(field, to_i2(local_pos), value); }

    template<typename T>
    void field_set(global T field[], int2 const local_pos, T const value) const {
        //printf("write %d, %d <- value %f\n", local_pos.x, local_pos.y, value);
        assert(in_bounds(local_pos));
        field[field_idx(local_pos)] = value;
    }

    float field_cached_get(local  BlockCache  const& cache
                          ,       float       const  default_value
                          ,global float       const  field[]
                          ,       int2        const  local_pos) const {
        //printf("fetch %d, %d (default %f)\n", local_pos.x, local_pos.y, default_value);
        assert(in_bounds_padded(local_pos));
        if (!in_bounds(local_pos))
            return field_get(default_value, field, local_pos);

        return cache[dim_idx(size, to_uc2(local_pos))];
    }

    void field_cached_set(local  BlockCache&       cache
                         ,global float             field[]
                         ,       int2        const local_pos
                         ,       float       const value) {
        //printf("write %d, %d = %f\n", local_pos.x, local_pos.y, value);
        assert(in_bounds_padded(local_pos));
        if (!in_bounds(local_pos))
            return field_set(field, local_pos, value);

        // compiler bug in AMD OCL impl: believes `value` could be of type __constant float or
        // __global float, which cannot be assigned to a __local float& (???)
        //cache[dim_idx(size, to_uc2(local_pos))] = value;
        cache.data[dim_idx(size, to_uc2(local_pos))] = value;
    }

    void cache_setup(global float const cost[], global float const time[]) {
        UTIL_FOREACH_WG_BGN(size, i) {
            uchar2 const pos = dim_pos(size, i);
            
#if ENABLE_CACHE_TIME
            // compiler bug in AMD OCL impl: believes `value` could be of type __constant float or
            // __global float, which cannot be assigned to a __local float& (???)
            //cache_time[i] = f;
            cache_time.data[i] = field_get(INFINITY, time, pos);
#endif
#if ENABLE_CACHE_COST
            cache_cost.data[i] = field_get(INFINITY, cost, pos);
#endif
        } UTIL_FOREACH_WG_END();
    }

    void cache_commit(global float time[]) {
        UTIL_FOREACH_WG_BGN(size, i) {
            uchar2 const pos = dim_pos(size, i);
#if ENABLE_CACHE_TIME
            field_set(time, pos, cache_time.data[i]);
#endif
        } UTIL_FOREACH_WG_END();
    }
};

// \hack workaround compiler bug in AMD OCL impl where it overgeneralises member functions by
//       qualifying their params with varying memory space qualifiers (e.g. global/local/etc)
//       even if these qualifiers are already specified.
//       Therefore, we declare plain ol' funcs instead.
float field_get_time(local  BlockInfo const& info
                    ,       float     const  default_value
                    ,global float     const  time[]
                    ,       int2      const  local_pos)
#if ENABLE_CACHE_TIME
{ return info.field_cached_get(info.cache_time, default_value, time, local_pos); }
#else
{ return info.field_get       (                 default_value, time, local_pos); }
#endif

void field_set_time(local  BlockInfo&       info
                   ,global float            time[]
                   ,       int2      const  local_pos
                   ,       float     const  value)
#if ENABLE_CACHE_TIME
{ info.field_cached_set(info.cache_time, time, local_pos, value); }
#else
{ info.field_set       (                 time, local_pos, value); }
#endif

float field_get_cost(local  BlockInfo const& info
                    ,       float     const  default_value
                    ,global float     const  cost[]
                    ,       int2      const  local_pos)
#if ENABLE_CACHE_COST
{ return info.field_cached_get(info.cache_cost, default_value, cost, local_pos); }
#else
{ return info.field_get       (                 default_value, cost, local_pos); }
#endif



// \note We're using a manhattan stencil.
typedef float EikonalAxisMin;

float mk_eikonal_axis_fetch(global float     const  time[]
                           ,local  BlockInfo const& info
                           ,       int2      const  pos) {
    // Out-of-cache should only occur when checking the outside-of-blocks for improved time.
    // In this case, just be pessimistic and assume inf.
    if (!info.in_bounds_padded(pos)) {
        //printf("out of padded");
        return INFINITY;
    }

    return field_get_time(info, INFINITY, time, pos);
}

EikonalAxisMin mk_eikonal_axis_min(global float     const  time[]
                                  ,local  BlockInfo const& info
                                  ,       int2      const  pos
                                  ,       int2      const  axis) {
    return min(mk_eikonal_axis_fetch(time, info, pos - axis)
              ,mk_eikonal_axis_fetch(time, info, pos + axis));
}

// ASSUME: Dimensionality is TINY (<= 3)
// We also want to minimise non-uniform flow.
// Conclusion: Bubble sort. (Uniform flow, 4 iterations for dim=3)
// template to poke/hint the compiler to consider length a constant and unroll
template<size_t K>
inline void eikonal_axis_sort(EikonalAxisMin (&ary)[K])
{ assert(K <= 3); return sort_bubble(ary); }


// Use the Godunov upwind difference scheme.
// See: "A FAST SWEEPING METHOD FOR EIKONAL EQUATIONS" by Zhao 2005 for an excellent overview of how
//      this works.
// \precondition `dim_sorted` is sorted. (duh. use eikonal_axis_sort if you must)
// \note we assume $h$ (aka the size of a cell) is baked into the cost field
// \note template to poke/hint the compiler to consider length a constant and unroll
template<size_t K>
float eikonal_solve(EikonalAxisMin const (&dim_sorted)[K]
                   ,float          const cost
                   ,float          const time_curr) {
    assert(0 <  K);
    assert(0 <= cost);
    assert(0 <= time_curr);
    // \todo we could bake this into the local cache instead...
    float const cost_sq = (cost * cost);

    // pretend: dim_K = inf
    // for i \in [0..K-1].
    //    let t = solve $sum_{j=0}^{i}{(x - dim_{j})^2} = cost^2$ for $x$
    //    if t <= dim_{i+1}.
    //      return t
    // 
    // sum_{j=0}^{i}{(x - dim_{j})^2} = cost^2
    // -> (i+1)*x^2 - x*2*(sum_{j=0}^{i}{dim_{j}}) + sum_{j=0}^{i}{dim_{j}^2} - cost^2 = 0

    float b         = 0;
    float c         = -cost_sq;
    float time_best = time_curr;
    for (size_t i = 0; i < K; ++i) {
        float time_axis = dim_sorted[i]; assert(0 <= time_axis);
        // can't improve by considering any later axis
        if (time_best <= time_axis  ) break;
        // if we start considering an INF term then there is no finite solution
        if (isinf(time_axis)        ) return INFINITY;

        // solve for lower quadratic (if real solution exists)
        b += -2        * time_axis;
        c += time_axis * time_axis;

        // 1st dimension solution looks slightly different than 1 <
        if (i == 0) {
            time_best = min(time_best, time_axis + 1 / cost);
            //printf("eikonal_solve 1st - curr=%f best=%f\n", time_curr, time_best);
        } else {
            float a    = i + 1;
            float t0   = b*b - 4*a*c;
            assert(0 <= t0);

            float t_a  = (0 <= t0) ? (-b - sqrt(t0)) / (2*a) : INFINITY;
            float t_b  = (0 <= t0) ? (-b + sqrt(t0)) / (2*a) : INFINITY;
            float t    = min(max(time_axis, t_a)
                            ,max(time_axis, t_b));
            //printf("eikonal_solve - a=%f b=%f c=%f -> %f, %f -> %f\n", a, b, c, t_a, t_b, t);
            assert(0 <= t); assert(t <= time_best);
            time_best  = t;
        }
    }

    return time_best;
}

#if 1
float eikonal_solve_2d(EikonalAxisMin const (&dim_sorted)[2]
                      ,float          const cost) {
    float const dX  = dim_sorted[0];
    float const dY  = dim_sorted[1];
    if (isinf(dX)) return cost + dY;
    if (isinf(dY)) return cost + dX;

    float const dXY = dX - dY;
    if (cost < fabs(dXY))
        return min(dX, dY) + cost;

    float const q = cost * cost * 2 - dXY * dXY; assert(0 <= q);
    float const p = (dX + dY + sqrt(q)) / 2;     assert(0 <= p);
    return p;
}

template<>
float eikonal_solve<2>(EikonalAxisMin const (&dim_sorted)[2]
                      ,float          const cost
                      ,float          const time_curr)
{ return min(time_curr, eikonal_solve_2d(dim_sorted, cost)); }
#endif





enum StoreMode { eStoreNever, eStoreIfBetter };
// \return the difference between the old and new time
float fim_block_2d_solve(       StoreMode  const  store_mode
                        ,global float      const  cost[]
                        ,global float             time[]
                        ,local  BlockInfo&        info
                        ,       int2       const  pos) {
    //atomic_inc(&info.solves);
    EikonalAxisMin dim[] = { mk_eikonal_axis_min(time, info, pos, (int2)(1, 0))
                           , mk_eikonal_axis_min(time, info, pos, (int2)(0, 1)) };
    /*if (isinf(dim[0]) && isinf(dim[1])) {
        if ((pos.x == 0) || pos.y == 0)
            printf("fim solve (%d, %d) dim = [%f, %f]\n", (int)pos.x, (int)pos.y, dim[0], dim[1]);
        return 0;
    }*/
    float  const cost_      = field_get_cost(info, 0.f, cost, pos);
    //printf("fim solve (%d, %d) cost = %f, dim = [%f, %f]\n", (int)pos.x, (int)pos.y, cost_, dim[0], dim[1]);
    eikonal_axis_sort(dim);
    //printf("\t-> [%f, %f]\n", dim[0], dim[1]);
    float  const time_old   = field_get_time(info, INFINITY, time, pos);
    float  const time_new   = eikonal_solve(dim, cost_, time_old);
    //printf("\t-> %f -> %f\n", time_old, time_new);
    assert(time_new <= time_old);
    switch (store_mode) {
    case eStoreNever    : break;
    case eStoreIfBetter : { // always better, since eikonal_solve returns time_old if < solution.
        //atomic_inc(&info.cells_updated);
        field_set_time(info, time, pos, time_new);
        assert(field_get_time(info, INFINITY, time, pos) == time_new);
    } break;
    }

    info.smallest_time[get_local_id(0)] = min(info.smallest_time[get_local_id(0)], time_new);

    return (time_new != time_old) ? time_new - time_old : 0;
}

// \return true iff converged
float fim_block_2d_solve_core(global float      const  cost[]
                             ,global float             time []
                             ,local  BlockInfo&        info
                             ,       uchar2     const  pos) {
    //info.field_set(time, pos, 0.f);
    return fim_block_2d_solve(eStoreIfBetter, cost, time, info, to_i2(pos));
}

enum Border { border_east = 0, border_north = 1, border_west = 2, border_south = 3, border_count__, border_first__ = border_east };

uint2 border_axis(Border const e) {
    switch (e) {
    case border_east : // FALL TO: west
    case border_west : return (uint2)(0, 1);
    case border_north: // FALL TO: south
    case border_south: return (uint2)(1, 0);
    }

    assert(false && "invalid border");
    return (uint2)(0);
}

int2 border_corner(uchar2 const block_size, Border const e) {
    // (0,0) is upper left
    switch (e) {
    case border_east : return (int2)(         - 1,            0);
    case border_west : return (int2)(block_size.x,            0);
    case border_north: return (int2)(           0,          - 1);
    case border_south: return (int2)(           0, block_size.y);
    }

    assert(false && "invalid border");
    return (int2)(0);
}

int2 border_block_offset(Border const e) {
    // (0,0) is upper left
    switch (e) {
    case border_east : return (int2)(-1,  0);
    case border_west : return (int2)( 1,  0);
    case border_north: return (int2)( 0, -1);
    case border_south: return (int2)( 0,  1);
    }

    assert(false && "invalid border");
    return (int2)(0);
}

bool check_border(       float      const  minimum_change
                 ,global float      const  cost[]
                 ,global float             time[]
                 ,local  bool            (&wg_voting_buffer)[WORKGROUP_SIZE]
                 ,local  BlockInfo&        block
                 ,       Border     const  border) {
    int2 const block_adj = to_i2(block.xy) + border_block_offset(border);
    if (!dim_in_bounds(to_i2(block.num), block_adj)) {
        //printf("(%d, %d) - %d, %d  outside of %d, %d\n", block.xy.x, block.xy.y, block_adj.x, block_adj.y, block.num.x, block.num.y);
        return false;
    }

     int2 const pos_base        = border_corner(block.size, border);
    uint2 const dir_unit        = border_axis  (            border);
    uint2 const block_sz_masked = to_ui2(block.size) * dir_unit;
    uint  const border_len      = block_sz_masked.x + block_sz_masked.y;
    //printf("border %d, border_len %u   dir %u, %u\n", border, border_len, dir_unit.x, dir_unit.y);
    bool improved = false;
    UTIL_FOREACH_WG_BGN(border_len, i) {
        int2  const local_pos = pos_base + to_i2(dir_unit) * i;
        float const time_diff = fim_block_2d_solve(eStoreIfBetter, cost, time, block, local_pos);
#ifdef  DEBUG_MSGS
        float const time_new  = field_get_time(block, INFINITY, time, local_pos);
        if (time_diff != 0)
            printf("(%d %d) border %d, pos %d, %d  fdiff -> %f (%f)\n", block.xy.x, block.xy.y, border, local_pos.x, local_pos.y, time_diff, time_new);
#endif
        improved |= time_diff <= -minimum_change;
    } UTIL_FOREACH_WG_END();
    //printf("border %d, len %u -> %d\n", border, border_len, improved ? 1 : 0);

    wg_voting_buffer[get_local_id(0)] = improved;
    barrier(CLK_LOCAL_MEM_FENCE); // commit/sync
    return workgroup_or_pwr_2(wg_voting_buffer);
}


// PRECONDITION: workgroup size must be pwr of 2 for list compaction
__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
kernel void fim_block_2d(          float   const minimum_change
                        ,global    float   const field_cost[/*field_size*/]
                        ,global    float         field_time[/*field_size*/]
                        ,          uint2   const field_size
                        ,global    BlockStatus        blocks_status[/*get_num_groups(0)*/]
                        ,global    BlockJob     const block_ids[/*get_num_groups(0)*/]
                        ,          uint         const are_seeds_available
                        ,constant  uchar2        block_open_list_initial[]
                        ,constant  uint    const block_open_list_initial_lengths[]
                        ) {
    local bool wg_voting_buffer[WORKGROUP_SIZE];
    local bool border_dirty_set[4]; // simplification: just dirty the entire border

    //printf("THIS IS BULLSHIT");
    assert(get_work_dim()    == 1               );
    assert(get_local_size(0) == WORKGROUP_SIZE  );

    local BlockInfo block;
    if (get_local_id(0) == 0) {
        block.solves        = 0;
        block.cells_updated = 0;

        block.id          = block_ids[get_group_id(0)].id;
        block.field_size  = field_size;
        block.num         = (field_size + uint2(BLOCK_SIZE_MAX - 1)) / uint2(BLOCK_SIZE_MAX);
        block.xy          = dim_pos(block.num, block.id);
        //printf("%d @ %d, %d of %d, %d\n", block.id, block.xy.x, block.xy.y, block.num.x, block.num.y);
        assert(block.id < dim_size(block.num));

        block.origin      = block.xy * BLOCK_SIZE_MAX;
        assert(all(block.origin < field_size));
        block.size        = to_uc2(min(field_size - block.origin, (uint)BLOCK_SIZE_MAX));
        /*
        printf("setup\n\tblock id %d\n"
               "\tblock pos %d, %d of %d, %d\n"
               "\tblock origin %d, %d\n"
               "\tblock_size %d, %d of %d, %d\n"
               "\tinitial open list len: [%d, %d)\n"
              ,block.id
              ,block.xy    .x, block.xy    .y, block.num .x, block.num .y
              ,block.origin.x, block.origin.y
              ,block.size  .x, block.size  .y, field_size.x, field_size.y
              ,open_list_idx_bgn, open_list_idx_end);
        assert(open_list_idx_bgn <= open_list_idx_end);
        assert((open_list_idx_end - open_list_idx_bgn) <= 2);
        for (size_t i = open_list_idx_bgn; i < open_list_idx_end; ++i) {
            printf("\t %d of %d: %d, %d\n", (int)(i+1), open_list_idx_end - open_list_idx_bgn
                  ,(int)block_open_list_initial[i].x, (int)block_open_list_initial[i].y);
        }
        //*/
    }

    block.smallest_time[get_local_id(0)] = INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    


    ////////////////////////////////
    // setup local caches and open_list
    ////////////////////////////////
    
    block.cache_setup(field_cost, field_time);
    /*
    if (get_local_id(0) == 0) {
        for (size_t y = 0; y < block.size.y; ++y) {
            for (size_t x = 0; x < block.size.x; ++x) {
                printf("%d - %d, %d = %f\n", block.id, x, y, field_get_cost(block, INFINITY, field_cost, (int2)(x, y)));
            }
        }
    }
    //*/

    barrier(CLK_LOCAL_MEM_FENCE); // commit everything

    ////////////////////////////////
    // FIM BODY
    ////////////////////////////////
    //printf("fim - begin main body %d\n", open_list_idx_end- open_list_idx_bgn);
    // No need for `barrier(CLK_LOCAL_MEM_FENCE);` to account for block cache fill b/c
    // fim_open_set_2d_compact does so to satisfy its own invariants.

    //*
    size_t const ary_len      = dim_size(block.size);
    size_t const worker_count = WORKGROUP_SIZE;
    size_t const passes       = (ary_len + worker_count - 1) / worker_count;
    do {
        bool all_converged = true;

        for (size_t i = 0; i < passes; ++i) {
            size_t idx = worker_count * i + get_local_id(0);
            if (idx < ary_len) {
                uchar2 const local_pos  = dim_pos(block.size, idx);
                //printf("%d -> %d, %d\n", aryjob_wg_info.worker_id, local_pos.x, local_pos.y);
                float  const time_diff  = fim_block_2d_solve_core(field_cost, field_time, block, local_pos);
                //printf("%f\n", time_diff);

                all_converged &= -minimum_change < time_diff;
            }

            barrier(CLK_LOCAL_MEM_FENCE); // commit/sync
        }
        //*/

        wg_voting_buffer[get_local_id(0)] = all_converged;
        barrier(CLK_LOCAL_MEM_FENCE); // commit/sync
    } while (!workgroup_and_pwr_2(wg_voting_buffer));
    //*/

    for (Border b = border_first__; b < border_count__; b = Border(b + 1))
        border_dirty_set[b] = check_border(minimum_change, field_cost, field_time, wg_voting_buffer, block, b);

    ////////////////////////////////
    // flush/commit local caches
    ////////////////////////////////

    block.cache_commit(field_time);

    global BlockStatus& status = blocks_status[get_group_id(0)];
    status.smallest_time = workgroup_min_pwr_2(block.smallest_time, WORKGROUP_SIZE);

    if (get_local_id(0) == 0) {
        status.solves        = block.solves;
        status.cells_updated = block.cells_updated;
        //printf("block %04d, %04d # of solves: %d (%f per cell)\n", block.xy.x, block.xy.y, block.num_solves, ((float)block.num_solves) / (block.size.x * block.size.y));
        for (size_t i = 0; i < length_of(border_dirty_set); ++i)
            status.border[i] = border_dirty_set[i];
    }
}

