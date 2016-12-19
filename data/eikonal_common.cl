
#include "util_common.cl"


#ifndef ENABLE_CACHE_TIME
#define ENABLE_CACHE_TIME 0
#endif

#ifndef ENABLE_CACHE_COST
#define ENABLE_CACHE_COST 0
#endif

#ifndef BLOCK_SIZE_MAX
    #define BLOCK_SIZE_MAX 32
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


typedef LocalAryChecked<float, BLOCK_SIZE_MAX * BLOCK_SIZE_MAX> BlockCache;

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