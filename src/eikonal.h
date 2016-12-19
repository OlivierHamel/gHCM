#pragma once

#include "util_field.h"
#include "util_dim.h"


using namespace glm;
using  int2  = glm::i32vec2;
using uint2  = glm::u32vec2;
using uchar2 = glm::u8vec2;

struct HcmConfig {
    enum class Kernel    { Systolic, FIM };
    enum class CacheMode { None, Time, Cost, Both };

    // `block_size` by `block_size`. u8 to deliberately limit it to something sane.
    u8          block_size              = 8;
    // % of overload vs. reported preferred workgroup size; e.g. 2 -> 200%, .5 -> 50%
    float       workers_overload        = 1.f;
    // % of overload vs. reported # of compute units. recommended it be 1 < due to latency hiding.
    // set absurdly high to just dump everything at once.
    float       compute_unit_overload   = 1.f;
    // consider a cell to be converged if it changes by < specific threshold.
    float       fim_minimum_change      = 1e-5f;
    CacheMode   cache_mode              = CacheMode::Both;
    Kernel      kernel                  = Kernel::Systolic;
};
std::ostream& operator<<(std::ostream& os, HcmConfig const& di);


struct EikonalResult {
    using duration_t = std::chrono::microseconds;

    Field2D<float>  field;
    duration_t duration_total, duration_device;
};


// defined in fim.cpp
EikonalResult eikonal_fmm_sequential(Field2D<float>     const& field_speed
                                    ,std::vector<uint2> const& seed_points);

// defined in fim.cpp
EikonalResult eikonal_fim_sequential(float              const  minimum_change
                                    ,Field2D<float>     const& field_speed
                                    ,std::vector<uint2> const& seed_points);

EikonalResult eikonal_hcm_opencl(cl::Device         const& device
                                ,HcmConfig          const& config
                                ,Field2D<float>     const& field_speed
                                ,std::vector<uint2> const& seed_points);

struct HcmSetup {
    HcmConfig         config;
    cl::Device        device;
    cl::Context       ctx;
    cl::CommandQueue  cmd_queue;
    cl::Program       program;
    cl::Program       program_dbg;
};

optional<HcmSetup> eikonal_hcm_opencl_setup(cl::Device const& device
                                           ,HcmConfig  const& config);

EikonalResult      eikonal_hcm_opencl_ex   (HcmSetup                  info
                                           ,Field2D<float>     const& field_speed
                                           ,std::vector<uint2> const& seed_points);


// \postcondition `dim`'s content is undefined
typedef float EikonalAxisMin;

template<size_t K>
float eikonal_solve(EikonalAxisMin const (&dim_sorted)[K], float const cost, float const best_prev) {
    static_assert(K < 10, "This hasn't been tested for K > 3..");

    assert(cost < INFINITY);

    float b         = 0;
    float c         = -(cost * cost);
    float time_best = best_prev;
    for (size_t i = 0; i < length_of(dim_sorted); ++i) {
        float time_axis = dim_sorted[i]; assert(0 <= time_axis);
        // can't improve by considering any later axis
        if (time_best <= time_axis  ) break;
        // if we start considering an INF term then there is no finite solution
        if (isinf(time_axis)        ) return time_best;

        // solve for lower quadratic (if real solution exists)
        b += -2        * time_axis;
        c += time_axis * time_axis;

        // 1st dimension solution looks slightly different than 1 <
        if (i == 0) {
            time_best = std::min(time_best, time_axis + cost);
            //printf("eikonal_solve 1st - curr=%f best=%f\n", time_curr, time_best);
        } else {
            float a    = float(i) + 1;
            float quad = b*b - 4*a*c; assert(0 <= quad);
#ifdef NDEBUG
            float t    = (-b - sqrt(quad)) / (2*a);
#else
            float t_a  = (-b - sqrt(quad)) / (2*a);
            float t_b  = (-b + sqrt(quad)) / (2*a);
            float t    = max(t_a, t_b);
            assert(min(t_a, t_b) <= time_axis);
            assert(time_axis     <  t);
#endif
            //float t = t_a;
            //printf("eikonal_solve - a=%f b=%f c=%f -> %f, %f -> %f\n", a, b, c, t_a, t_b, t);
            // allow for slight drift due to precision (?)
            assert(0 <= t); assert(-.01 < time_best - t);
            time_best = min(t, time_best);
        }
    }

    return time_best;
}

#if 1
template<>
inline float eikonal_solve<2>(EikonalAxisMin const (&dim_sorted)[2]
                             ,float          const cost
                             ,float          const time_curr) {
    float const dX  = dim_sorted[0];
    float const dY  = dim_sorted[1];
    if (isinf(dX)) return cost + dY;
    if (isinf(dY)) return cost + dX;

    float const dXY = dX - dY;
    if (cost < fabs(dXY))
        return min(time_curr, min(dX, dY) + cost);

    float const q = cost * cost * 2 - dXY * dXY; assert(0 <= q);
    float const p = (dX + dY + sqrt(q)) / 2;     assert(0 <= p);
    return min(time_curr, p);
}
#endif


// \return a-b; inf - inf is considered (unsigned) 0 instead of NaN
inline float eikonal_time_diff(float a, float b)
{ return (a == b) ? 0 : (a - b); }

