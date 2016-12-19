
#pragma OPENCL EXTENSION cl_amd_printf                  : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store  : enable

#include "eikonal_common.cl"


struct FIM_OpenSet2d {
    // identity map listing of cells to add to open
    //   i.e. open[blk x * y + x] == true  <-> xy is in the set of open items.
    // \note See: `set_generation`
    LocalAryChecked<bool  , BLOCK_SIZE_MAX * BLOCK_SIZE_MAX> ary; // 32 * 32 * 1 =  1     KiB
};

struct FIM_OpenList2d {
    // compacted list of open cells
    LocalAryChecked<uchar2, BLOCK_SIZE_MAX * BLOCK_SIZE_MAX> ary; // 32 * 32 * 2 =  2     KiB
    size_t  length;
};

void fim_open_set_2d_clear(local FIM_OpenSet2d*  const set
                          ,      uchar2          const block_size) {
    //printf("fim_open_set_2d_clear - block_size: %d, %d\n", (int)block_size.x, (int)block_size.y);
    UTIL_FOREACH_WG_BGN(block_size, elem_idx) {
        set->ary[elem_idx] = false;
    } UTIL_FOREACH_WG_END();
}

void fim_open_set_2d_init(local FIM_OpenSet2d*  const set
                         ,      uchar2          const block_size)
{ fim_open_set_2d_clear(set, block_size); }

void fim_open_set_2d_add(local FIM_OpenSet2d* const set
                        ,      uchar2         const block_size
                        ,      uchar2         const local_pos)
{ set->ary[dim_idx(to_ui2(block_size), to_ui2(local_pos))] = true; }

void fim_open_set_2d_remove(local FIM_OpenSet2d* const set
                           ,      uchar2         const block_size
                           ,      uchar2         const local_pos)
{ set->ary[dim_idx(to_ui2(block_size), to_ui2(local_pos))] = false; }

// PRECONDITION: syncronised w/ LOCAL_MEM
bool fim_open_set_2d_contains(local FIM_OpenSet2d const*  const set
                             ,      uchar2                const block_size
                             ,      uchar2                const local_pos)
{ return all(local_pos < block_size)
      && set->ary[dim_idx(to_ui2(block_size), to_ui2(local_pos))]; }

// PRECONDITIONS: invocation must be uniform as this function `barrier`s on LOCAL_MEM.
void fim_open_set_2d_compact(local ushort                     wg_accum_buffer[/*wg size * uint*/]
                            ,local FIM_OpenSet2d const* const set
                            ,      uchar2               const block_size
                            ,local FIM_OpenList2d*      const out_list) {
    /*
    size_t num_brute_force_calc = 0;
    if (get_local_id(0) == 0) {
        for (size_t y = 0; y < block_size.y; ++y) {
            for (size_t x = 0; x < block_size.x; ++x) {
                num_brute_force_calc += fim_open_set_2d_contains(set, block_size, (uchar2)(x, y)) ? 1 : 0;
            }
        }

        printf("\nFIM WAVE - open list: %d\n---------\n", (int)num_brute_force_calc);
        for (size_t y = 0; y < block_size.y; ++y) {
            printf("\t|");
            for (size_t x = 0; x < block_size.x; ++x) {
                printf(fim_open_set_2d_contains(set, block_size, (uchar2)(x, y)) ? "*" : "_");
            }
            printf("| %d\n", y);
        }
    }
    //*/

    // prefix-sum the # of slots each worker wants so we agree on the indices to use
    //   ~ 2 * lg(block_size.x * block_size.y) cost to do compaction
    // 
    // INVARIANT: block_size \in [0, d), where d is the number of neighbours for each cell
    ushort num_cells_to_add = 0;
    UTIL_FOREACH_WG_BGN(block_size, elem_idx) {
        if (fim_open_set_2d_contains(set, block_size, dim_pos(block_size, elem_idx))) {
            //printf("compact %d -> check %d, %d  = yes\n", (int)get_local_id(0)(), (int)dim_pos(block_size, elem_idx).x, (int)dim_pos(block_size, elem_idx).y);
            num_cells_to_add++;
        }
    } UTIL_FOREACH_WG_END();
    //if (0 < num_cells_to_add) printf("compact %d -> %d\n", (int)get_local_id(0)(), (int)num_cells_to_add);
    wg_accum_buffer[get_local_id(0)] = num_cells_to_add;
    barrier(CLK_LOCAL_MEM_FENCE); // commit changes to wg_accum_buffer
    /*
    if (worker_id == 0) {
        for (int i = 0; i < wg_size; ++i)
            printf("%d\t", wg_accum_buffer[i]);

        printf("\n");
    }
    */
    barrier(CLK_LOCAL_MEM_FENCE);
    workgroup_scan_pwr_2(wg_accum_buffer, WORKGROUP_SIZE);
    // no commit necessary post `workgroup_scan_pwr_2` invoke
    /*
    if (worker_id == 0) {
        printf("->\n");
        for (int i = 0; i < wg_size; ++i)
            printf("%d\t", wg_accum_buffer[i]);

        printf("\n");
    }
    */
    barrier(CLK_LOCAL_MEM_FENCE);

    out_list->length = wg_accum_buffer[WORKGROUP_SIZE - 1];
    //if (get_local_id(0) == 0) assert(out_list->length == num_brute_force_calc);
    if (out_list->length == 0) return;

    // \todo \fixme Buuuuuullshit-ish b/c we keep populating the set/list in the same order.
    // Possible repercussions on convergence soeed?
    // 
    // Alright, we're ready to populate the list.
    // Rerun through the set, and use num_cells_to_add as our induction variable.
    // Note: wg_accum_buffer[worker_id] = num_cells_to_add + <everything before worker_id> since we
    //       prefix summed it.
    ushort const last_index_alloced = wg_accum_buffer[get_local_id(0)];
    UTIL_FOREACH_WG_BGN(block_size, elem_idx) {
        size_t const list_idx  = last_index_alloced - num_cells_to_add;
        uchar2 const local_pos = dim_pos(block_size, elem_idx);

        if (fim_open_set_2d_contains(set, block_size, local_pos)) {
            assert(0 < num_cells_to_add);
            num_cells_to_add        -= 1;
            out_list->ary[list_idx]  = local_pos;
        }
    } UTIL_FOREACH_WG_END();
    barrier(CLK_LOCAL_MEM_FENCE); // commit `out_list->ary` changes
}


/*
bool fim_block_2d_solve_neighbour_could_improve
                                 (global float          const  cost[]
                                 ,global float                 time[]
                                 ,local  BlockInfo&            info
                                 ,       uchar2         const  center
                                 ,        char2         const  offset) {
    
    int2  const pos_i     = to_i2(center) + to_i2(offset);
    if (!info.in_bounds_field(to_i2(info.origin) + pos_i)) return false;
    //printf("check %d, %d\n", pos_i.x, pos_i.y);
    float const pos_cost  = info.field_get(INFINITY, cost, pos_i );
    float const pos_time  = info.field_get_time(INFINITY, time, pos_i );
    float const ctr_time  = info.field_get_time(INFINITY, time, to_i2(center));
    if ((ctr_time + pos_cost) <= pos_time) {
        //printf("improve possible %d, %d -> %d, %d due to min cost\n", center.x, center.y, pos_i.x, pos_i.y);
        return true; // absolutely could improve
    }

    float const pos2_time = info.field_get_time(INFINITY, time, pos_i + to_i2(offset));
    if (ctr_time < pos2_time) {
        //printf("improve possible %d, %d < %d, %d opposite neighbour\n", center.x, center.y, (pos_i + to_i2(offset)).x, (pos_i + to_i2(offset)).y);
        return true;
    }

    return false;
}
*/

void fim_block_2d_solve_neighbour(float   const minimum_change
                                 ,local  bool                  border_dirty_set[/*4*/]
                                 ,local  FIM_OpenSet2d* const  open_set
                                 ,global float          const  cost[]
                                 ,global float                 time[]
                                 ,local  BlockInfo&            info
                                 ,       uchar2         const  center
                                 ,        char2         const  offset) {
    int2 const pos_i = to_i2(center) + to_i2(offset);
    if (!info.in_bounds(pos_i)) {
        int const border = (        pos_i.x  <  0      ) ? 0 : 
                           (        pos_i.y  <  0      ) ? 1 :
                           (int(info.size.x) <= pos_i.x) ? 2 :
                                                           3;
        //if (info.field_get(INFINITY, time, pos_i) <= 0) return;
        float const diff = fim_block_2d_solve(eStoreNever, cost, time, info, pos_i);
        bool const dirtify = diff < -minimum_change;
        //bool const dirtify = fim_block_2d_solve_neighbour_could_improve(cost, time, info, center, offset);
        /*if (bad_block && dirtify) {
            printf("%d, %d -> diff = %f\n", pos_i.x, pos_i.y, diff);
        }*/
        /*if (dirtify) {
            printf("(%d, %d) %d, %d -> diff = %f (border %d\n", info.xy.x, info.xy.y, pos_i.x, pos_i.y, diff, border);
        }*/
        border_dirty_set[border] = border_dirty_set[border]
                                || dirtify;
        //printf("out of block: %d, %d = %d  (%f)\n", pos_i.x, pos_i.y, (int)border_dirty_set[border], diff);
        return;
    }

    //printf("in block: %d, %d\n", pos_i.x, pos_i.y);
    uchar2 const pos = to_uc2(pos_i);
    if (fim_open_set_2d_contains(open_set, info.size, pos))  {
        //printf("\tAlready in open.\n");
        return;
    }

    //if (info.field_get(INFINITY, time, pos_i) <= 0) return;
    //float  const time_diff = info.field_get(time, pos, time_new);//
    float  const time_diff = fim_block_2d_solve(eStoreIfBetter, cost, time, info, pos_i);
    bool   const improved  = time_diff < -minimum_change;
    //bool const improved = fim_block_2d_solve_neighbour_could_improve(cost, time, info, center, offset);
    //printf("\tTime Diff: %f\n", time_diff);
    if (improved) fim_open_set_2d_add(open_set, info.size, pos);
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

    local ushort         workgroup_scratchpad[WORKGROUP_SIZE];
    local bool           border_dirty_set    [4]; // simplification: just dirty the entire border
    local FIM_OpenSet2d  open_set;      //                       1     KiB
    local FIM_OpenList2d open_list;     //                       2     KiB
                                        //                      ~3     KiB
    local BlockInfo      block;

    uint const open_list_idx_bgn = are_seeds_available ? ((get_group_id(0) == 0) ? 0 : block_open_list_initial_lengths[get_group_id(0) - 1]) : 0;
    uint const open_list_idx_end = are_seeds_available ?                               block_open_list_initial_lengths[get_group_id(0)    ]  : 0;

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

        for (int i = 0; i < length_of(border_dirty_set); ++i)
            border_dirty_set[i] = false;
    }

    block.smallest_time[get_local_id(0)] = INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    


    ////////////////////////////////
    // setup local caches and open_list
    ////////////////////////////////
    //printf("setup - initing open_set\n");
    fim_open_set_2d_init(&open_set, block.size);
    //printf("setup - initing inital open list\n");
    UTIL_FOREACH_WG_BGN(size_t(open_list_idx_end - open_list_idx_bgn), i) {
        fim_open_set_2d_add(&open_set, block.size, block_open_list_initial[open_list_idx_bgn + i]);
    } UTIL_FOREACH_WG_END();

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

    //printf("Border Dirty: 0x%08x\n", border_dirty);
    // border 0: x-
    uint const border_dirty = block_ids[get_group_id(0)].border;
    if (border_dirty & 0xFF) {
        UTIL_FOREACH_WG_BGN(size_t(block.size.y), i) {
            fim_open_set_2d_add(&open_set, block.size, (uchar2)(0, i));
        } UTIL_FOREACH_WG_END();
    }

    // border 2: x+
    if (border_dirty & 0xFF0000) {
        UTIL_FOREACH_WG_BGN(size_t(block.size.y), i) {
            fim_open_set_2d_add(&open_set, block.size, (uchar2)(block.size.x-1, i));
        } UTIL_FOREACH_WG_END();
    }

    // border 1: y-
    if (border_dirty & 0xFF00) {
        UTIL_FOREACH_WG_BGN(size_t(block.size.x), i) {
            fim_open_set_2d_add(&open_set, block.size, (uchar2)(i, 0));
        } UTIL_FOREACH_WG_END();
    }

    // border 3: y-
    if (border_dirty & 0xFF000000) {
        UTIL_FOREACH_WG_BGN(size_t(block.size.x), i) {
            fim_open_set_2d_add(&open_set, block.size, (uchar2)(i, block.size.y-1));
        } UTIL_FOREACH_WG_END();
    }
    
    barrier(CLK_LOCAL_MEM_FENCE); // commit everything
    for (size_t i = open_list_idx_bgn; i < open_list_idx_end; ++i)
        assert(fim_open_set_2d_contains(&open_set, block.size, block_open_list_initial[i]));


    ////////////////////////////////
    // FIM BODY
    ////////////////////////////////
    //printf("fim - begin main body %d\n", open_list_idx_end- open_list_idx_bgn);
    // No need for `barrier(CLK_LOCAL_MEM_FENCE);` to account for block cache fill b/c
    // fim_open_set_2d_compact does so to satisfy its own invariants.

    //*
    for (;;) {
        // INVARIANT: list can only contain at most one instance of a given block pos.
        fim_open_set_2d_compact(workgroup_scratchpad, &open_set, block.size, &open_list);
        if (open_list.length == 0) break;
        //*

        UTIL_FOREACH_WG_BGN(open_list.length, elem_idx) {
            uchar2 const local_pos  = open_list.ary[elem_idx];
            //printf("%d -> %d, %d\n", aryjob_wg_info.worker_id, local_pos.x, local_pos.y);
            float  const time_diff  = fim_block_2d_solve_core(field_cost, field_time, block
                                                             ,local_pos);
            //printf("%f\n", time_diff);
            if (-minimum_change < time_diff) {
                // no need to `barrier` time update before checking neighbours b/c we ignore neighbours
                // who are in the open list.
                fim_block_2d_solve_neighbour(minimum_change, border_dirty_set, &open_set, field_cost, field_time, block, local_pos, (char2)( 0, -1));
                fim_block_2d_solve_neighbour(minimum_change, border_dirty_set, &open_set, field_cost, field_time, block, local_pos, (char2)( 0,  1));
                fim_block_2d_solve_neighbour(minimum_change, border_dirty_set, &open_set, field_cost, field_time, block, local_pos, (char2)(-1,  0));
                fim_block_2d_solve_neighbour(minimum_change, border_dirty_set, &open_set, field_cost, field_time, block, local_pos, (char2)( 1,  0));

                fim_open_set_2d_remove(&open_set, block.size, local_pos);
            }
        } UTIL_FOREACH_WG_END();
        //*/
        barrier(CLK_LOCAL_MEM_FENCE); // commit/sync
    }
    //*/


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

