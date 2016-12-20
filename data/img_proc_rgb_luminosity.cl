
#include "util_common.cl"


kernel void img_proc_rgb_luminosity(global float const input [/*3 * size.x * size.y*/]
                                   ,global float       output[/*    size.x * size.y*/]
                                   ,       uint2 const img_size
                                   ) {
    assert(get_work_dim()       == 1                  );
    assert(get_global_size(0)   == dim_size(img_size) );

    size_t const idx                  = get_global_id(0);
    assert(idx                  <  dim_size(img_size) );

    float3 const k_rgb_to_luminosity  = (float3)(0.299, 0.587, 0.114);
    float3 const rgb                  = (float3)(input[idx * 3 + 0]
                                                ,input[idx * 3 + 1]
                                                ,input[idx * 3 + 2]);
    output[idx] = dot(rgb, k_rgb_to_luminosity);
}

