
#pragma OPENCL EXTENSION cl_amd_printf                  : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store  : enable

#include "eikonal_common.cl"


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
    assert(         get_work_dim()      == 1              );
    assert(         get_local_size(0)   == WORKGROUP_SIZE );
    // workgroup should be pwr of 2 due to our scan/min/and/etc..
    assert(popcount(get_local_size(0))  == 1              );

    local bool      wg_voting_buffer[WORKGROUP_SIZE];
    local bool      border_dirty_set[4]; // simplification: just dirty the entire border
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

