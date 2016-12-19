
#include "eikonal.h"
#include "util_opencl.h"
#include "bitset_dynamic.h"
#include "util_files.h"


namespace {

// CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE is only avail once we've compiled.
// can't compile until we know the preferred group size.
//  -> cheat. on a r280x it seems its always 64 for our purposes.
auto const k_hack_device_pref_wg_size   = 64u;
auto const k_hcm_kernel_file_systolic   = "data/eikonal_ghcm_systolic";
auto const k_hcm_kernel_file_fim        = "data/eikonal_ghcm_fim";

struct Border { enum Type { East, North, West, South }; };
int2 const k_border_directions[] { int2(-1, 0), int2(0, -1), int2(1, 0), int2(0, 1) };

using BlockId = cl_uint;

struct BlockJob {
    BlockId id;
    cl_uint border;
};

struct BlockStatus {
    cl_uint solves;
    cl_uint cells_updated;
    float   smallest_time = INFINITY;
    cl_uint border[4];
};

struct BlocksInfo {
    std::vector<BlockStatus> status;
    uint2                    size;
};

struct ProfilingInfo {
    BlockId block_id;
    size_t  device_time_ns;
    size_t  total_solves;
    size_t  total_improved;
};

// \hack this is ugly as hell. clean it up/replace w/ better data-flow
struct BlockWrapper {
    BlocksInfo* host;
    BlockId     id;
    float       recorded_min;

    BlockWrapper() = default;
    BlockWrapper(BlocksInfo* h, BlockId i) : host(h), id(i)
    { recorded_min = calc_min_time(); }

    BlockWrapper(BlockWrapper const&) = default;

    float calc_min_time() const {
        assert(host);
        auto const center = int2(dim_pos(host->size, id));
        float      t      = host->status[id].smallest_time;
        //*
        for (auto&& dir : k_border_directions) {
            auto const pos = center + dir;
            if (dim_in_bounds(int2(host->size), pos))
                t = std::min(t, host->status[dim_idx(host->size, uint2(pos))].smallest_time);
        }
        //*/
        return t;
    }

    BlockId block_id() const { assert(host); return id; }

    // \todo \fixme this is imprecise. ideally we'd be tracking the exact provoking min
    //              from the borders
    bool operator<(BlockWrapper const& o) const {
        assert(host); assert(o.host); assert(host == o.host);
        return recorded_min > o.recorded_min;
    }
};


cl_uint2 to_cl_ui2(uint2 const v) { cl_uint2 a; a.x = v.x; a.y = v.y; return a; };


void CL_CALLBACK cl_notify_func(const char* const err_info
                               ,const void* const /*private_data*/
                               ,size_t      const /*user_data_count_bytes*/
                               ,void*       const /*user_data*/) {
    std::cout << "CL Notify: " << err_info << "\n";
}

std::string hcm_build_opts(cl::Device const& device
                          ,HcmConfig  const& config) {
    auto const cache_mode_on = [&](HcmConfig::CacheMode const e) {
        auto const on = (config.cache_mode == e                         )
                     || (config.cache_mode == HcmConfig::CacheMode::Both);
        return on ? 1 : 0;
    };

    std::stringstream build_opt;
    build_opt << " -x clc++ -cl-mad-enable"
              << " -D BLOCK_SIZE_MAX="
                << max<u32>(1u, config.block_size)
              << " -D WORKGROUP_SIZE="
                << max<u32>(1u, u32(config.workers_overload * k_hack_device_pref_wg_size))
              << " -D ENABLE_CACHE_COST=" << cache_mode_on(HcmConfig::CacheMode::Cost)
              << " -D ENABLE_CACHE_TIME=" << cache_mode_on(HcmConfig::CacheMode::Time);

    if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU)
        build_opt << " -D DEVICE_IS_CPU";

    return build_opt.str();
}

}

std::ostream& operator<<(std::ostream& os, HcmConfig const& config) {
    auto const mode_name = ([&] {
        switch (config.cache_mode) {
            case HcmConfig::CacheMode::None: return "None";
            case HcmConfig::CacheMode::Time: return "Time";
            case HcmConfig::CacheMode::Cost: return "Cost";
            case HcmConfig::CacheMode::Both: return "Time & Cost";
        }
        assert(false);
        return "unknown";
    })();

    os  <<   "FIM Threshold         : " << config.fim_minimum_change
        << "\nBlock Size            : " << (size_t)config.block_size
        << "\nWorker Overload       : " << config.workers_overload
        << "\nCompute Unit Overload : " << config.compute_unit_overload
        << "\nCache Mode            : " << mode_name
        << "\n";
    return os;
}

optional<HcmSetup> eikonal_hcm_opencl_setup(cl::Device const& device
                                           ,HcmConfig  const& config) {
    auto const ocl_file = ([&] {
        switch (config.kernel) {
        case HcmConfig::Kernel::Systolic: return k_hcm_kernel_file_systolic;
        case HcmConfig::Kernel::FIM     : return k_hcm_kernel_file_fim;
        }
        assert(false && "Unreachable");
        return "";
    })();

    try {
        HcmSetup info {};
        info.config     = config;
        info.device     = device;
        info.ctx        = cl::Context(device, nullptr, cl_notify_func);
        info.cmd_queue  = cl::CommandQueue(info.ctx, device, CL_QUEUE_PROFILING_ENABLE);

        auto const program      = cl_util_build(info.ctx, { device }
                                               ,hcm_build_opts(device, config).c_str()
                                               ,ocl_file);
        auto const program_dbg  = cl_util_build(info.ctx, { device }
                                               ,(hcm_build_opts(device, config) + " -D DEBUG_MSGS").c_str()
                                               ,ocl_file);
        if (!(program && program_dbg)) {
            std::cout << "Failed to build program: " << ocl_file << "\n";
            return {};
        }

        info.program     = *program;
        info.program_dbg = *program_dbg;
        return info;
    } catch (const cl::Error& e) {
        auto const who = e.what() ? e.what() : "<null>";
        auto const err = cl_error_message(e.err());
        std::cout << "OpenCL Error.\n\tTask: gHCM Setup\n\tWho: " << who << "\n\tErr: " << err << "\n";
        return {};
    }
}

EikonalResult eikonal_hcm_opencl(cl::Device         const& device
                                ,HcmConfig          const& config
                                ,Field2D<float>     const& field_cost
                                ,std::vector<uint2> const& seed_points) {
    auto const info = eikonal_hcm_opencl_setup(device, config); assert(info);
    if (!info) { getchar(); exit(-1); }
    return eikonal_hcm_opencl_ex(*info, field_cost, seed_points);
}

EikonalResult eikonal_hcm_opencl_ex(HcmSetup                  info
                                   ,Field2D<float>     const& field_cost
                                   ,std::vector<uint2> const& seed_points) {
    auto       field_time     = Field2D<float>(field_cost.size(), INFINITY);
    for (auto&& s : seed_points)
        field_time[s] = 0;

    auto const wg_size_max    = max<u32>(1u, u32(info.config.workers_overload * k_hack_device_pref_wg_size));
    auto const block_size_max = uint2(max<u8>(1u, info.config.block_size));
    auto const blocks         = (field_cost.size() + block_size_max - uint2(1)) / block_size_max;
    auto const num_blocks     = blocks.x * blocks.y; assert(0 < num_blocks);

    std::multimap<BlockId, uchar2> seed_map = ([&] {
        // \fixme \todo be less naive about bucketing these. for now this is fine since we only have
        // a tiny number of seed points
        std::multimap<BlockId, uchar2> map;
        for (auto&& p : seed_points) {
            for (auto&& dir : k_border_directions) {
                auto const pos          = int2(p) + dir;
                if (!dim_in_bounds(int2(field_time.size()), pos)) continue;

                auto const block_xy     = uint2(pos) / block_size_max;
                auto const block_origin = block_xy   * block_size_max;
                auto const block_id     = (BlockId)dim_idx(blocks, block_xy);
                map.insert(std::make_pair(block_id, uchar2(uint2(pos) - block_origin)));
            }
        }

        return map;
    })(); // enqueue blocks w/ seeds

    //std::map   <BlockId, size_t> explored_blocks;
    std::vector<ProfilingInfo > profiling_info;
    std::vector<BlockId       > blocks_border_mask    (num_blocks);
    BlocksInfo                  block_infos;
    block_infos.size = blocks;
    block_infos.status.resize(num_blocks);

    auto blocks_dirty_set   = BitsetDynamic(num_blocks);
    auto blocks_dirty_queue = ([&] {
        std::priority_queue<BlockWrapper> queue;

        for(auto&& kv : seed_map) {
            if (blocks_dirty_set[kv.first]) continue; // already got it
            blocks_dirty_set[kv.first] = true;

            auto& block_status = block_infos.status[kv.first];
            block_status.smallest_time = 0;

            queue.push(BlockWrapper { &block_infos, kv.first });
        }
        return queue;
    })();


    auto const k_dump_interval = 100;
    size_t num_wave = 0;
    auto const dump_visualisation = [&](std::priority_queue<BlockWrapper> const& q, BlockJob const disp[], size_t const disp_len) {
#if 0
        if (((num_wave % k_dump_interval) == 0)
         && (num_wave < k_dump_interval * 20)) {
            Field2D<float> f(blocks, INFINITY);
        
            for (auto cpy = q; !cpy.empty(); cpy.pop()) f[cpy.top().id] = 4;
            for (size_t i = 0; i < disp_len; ++i      ) f[disp[i]  .id] = 0;
            std::cout << "Disp " << disp_len << "; Remaining: " << q.size() << "\n";

            std::stringstream ss;
            ss << "data/dbg_wave_" << num_wave << "_active.png";
            file_write_time_field(ss.str().c_str(), f, 0, 5);

            Field2D<float> g(blocks);
            for (size_t i = 0; i < block_infos.status.size(); ++i)
                g[i] = block_infos.status[i].smallest_time;
            
            std::stringstream ss2;
            ss2 << "data/dbg_wave_" << num_wave << "_smallest_time.png";
            file_write_time_field(ss2.str().c_str(), g);
        }
#endif
        num_wave++;
    };

    EikonalResult result;
    cl_do("run eikonal fim block kernel", [&] {
        auto const whole_bgn = std::chrono::high_resolution_clock::now();
        /*
        32 KiB local
        3  GiB global

        63 KiB constant buffer

        Pref Wg Size: 64    ( 8^2 block?)
        Max  Wg Size: 256   (16^2 block?)

        kernel void fim_block_2d(          float   const minimum_change
                        ,global    float   const field_cost[/*field_size* /]
                        ,global    float         field_time[/*field_size* /]
                        ,          uint2   const field_size
                        ,global    BlockStatus        blocks_status[/* # of blocks * /]
                        ,global    BlockJob     const block_ids[/*get_num_groups(0)* /]
                        ,          uint         const are_seeds_available
                        ,constant  uchar2        block_open_list_initial[]
                        ,constant  uint    const block_open_list_initial_lengths[]
                        ,          uint    const do_debug
                        ) {
        */

        auto const  prefWgSizeMul = cl::Kernel(info.program, "fim_block_2d")
            .getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(info.device);
        std::cout   << "Pref Wg Size: " << prefWgSizeMul                << "\n";

        auto       kernel     = cl::KernelFunctor<cl_float      /* min change   */
                                                 ,cl::Buffer    /* cost         */
                                                 ,cl::Buffer    /* time         */
                                                 ,cl_uint2      /* field size   */
                                                 ,cl::Buffer    /* status       */
                                                 ,cl::Buffer    /* jobs         */
                                                 ,cl_uint       /* are_seeds_available */
                                                 ,cl::Buffer    /* open vals    */
                                                 ,cl::Buffer    /* open scan    */
                                                 >(info.program, "fim_block_2d");
        auto       kernel_dbg     = cl::KernelFunctor<cl_float      /* min change   */
                                                 ,cl::Buffer    /* cost         */
                                                 ,cl::Buffer    /* time         */
                                                 ,cl_uint2      /* field size   */
                                                 ,cl::Buffer    /* status       */
                                                 ,cl::Buffer    /* jobs         */
                                                 ,cl_uint       /* are_seeds_available */
                                                 ,cl::Buffer    /* open vals    */
                                                 ,cl::Buffer    /* open scan    */
                                                 >(info.program_dbg, "fim_block_2d");

        
        auto const buf_cost     = cl::Buffer(info.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR
                                            ,field_cost.size_flat() * sizeof(float)
                                            ,const_cast<float*>(field_cost.backing_store().data()));
        auto const buf_time     = cl::Buffer(info.ctx, field_time.backing_store().begin()
                                            ,field_time.backing_store().end(), false, true );
        auto const buffer_empty = cl::Buffer(info.ctx, CL_MEM_READ_ONLY, sizeof(cl_uchar2));
        
        auto const buf_open_lens_empty = ([&] {
            std::vector<cl_uint> tmp(wg_size_max, 0);
            return cl::Buffer(info.ctx, tmp.begin(), tmp.end(), true, false);
        })();


        auto const units_avail  = info.device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        auto const max_jobs     = max<size_t>(1u, size_t(info.config.compute_unit_overload * units_avail));
        std::vector<BlockJob    > job_ids;    job_ids   .resize(max_jobs);
        std::vector<BlockStatus > job_result; job_result.resize(max_jobs);

        auto const buf_jobs     = cl::Buffer(info.ctx, job_ids   .begin(), job_ids   .end()
                                            ,true , true);
        auto const buf_status   = cl::Buffer(info.ctx, job_result.begin(), job_result.end()
                                            ,false, true);

        while (!blocks_dirty_queue.empty()) {
            auto const num_jobs     = ([&] {
                size_t jobs = 0;
                for (size_t i = 0; i < max_jobs; ++i) {
                    if (blocks_dirty_queue.empty()) break;

                    auto const id = blocks_dirty_queue.top().block_id();
                    job_ids   [i] = { id, blocks_border_mask[id] };
                    job_result[i] = block_infos.status[id];
                    blocks_dirty_queue.pop();
                    blocks_border_mask[id] = 0;
                    jobs++;
                }

                return jobs;
            })();
            if (max_jobs < num_jobs) __debugbreak();
            dump_visualisation(blocks_dirty_queue, job_ids.data(), num_jobs);
            /*
            if (blocks_to_dispatch.size() < 10) {
            std::cout << "dispatching blocks (" << blocks_to_dispatch.size() << "):\n";
            for (auto&& job : blocks_to_dispatch)
                std::cout << "\tBlock: " << dim_pos(blocks, job.id).x << ", " << dim_pos(blocks, job.id).y << block_infos.status[job.id].smallest_time << "\n";
            }
            //*/

#if 0
            std::cout << "hcm new block set (length " << blocks_dirty_queue.size() << ")\n";

            std::vector<char> visualisation(blocks.x * blocks.y, '_');
            for (auto&& b : blocks_to_dispatch)
                visualisation[b.id] = '*';

            for (size_t y = 0; y < blocks.y; ++y) {
                std::cout << "\t|";
                for (size_t x = 0; x < blocks.x; ++x)
                    std::cout << visualisation[dim_idx(blocks, {x, y})];
                std::cout << "| " << y << "\n";
                //auto const block_xy = uint2(block_id % blocks.y, block_id / blocks.y);
                //auto const block_origin = block_xy * k_hcm_block_size;
                //std::cout << "hcm doing block: " << block_xy.x << ", " << block_xy.y << "\n";
            }
#endif

            

            std::vector<cl_uchar2> open_list_vals;
            std::vector<cl_uint  > open_list_sizes;
            bool do_debug = false;
            cl_uint job_id_min = (cl_uint)dim_size(blocks);
            cl_uint job_id_max = 0;
            for (size_t i = 0; i < num_jobs; ++i) {
                auto const&job          = job_ids[i];
                auto const block_xy     = dim_pos(blocks, job.id);
                auto const block_origin = block_xy * block_size_max;
                //explored_blocks[job.id]++;
                job_id_min = std::min(job_id_min, job.id);
                job_id_max = std::max(job_id_max, job.id);
                //if (explored_blocks[job.id] > 50) {
                //    //std::cout << "Excessive re-processing " << block_xy.x << ", " << block_xy.y << ": " << explored_blocks[job.id] << "\n";
                //    do_debug = (num_jobs < 10);
                //}

                auto const iter_pair = seed_map.equal_range(job.id);
                if (iter_pair.first == iter_pair.second) continue;

                
                for (auto iter = iter_pair.first; iter != iter_pair.second; iter = seed_map.erase(iter)) {
                    assert(dim_in_bounds(block_size_max, uint2(iter->second)));

                    cl_uchar2 xy; xy.x = (*iter).second.x; xy.y = (*iter).second.y;
                    open_list_vals.push_back(xy);
                }
                open_list_sizes.push_back(cl_uint(open_list_vals.size()));
            }

            // in OCL, size zero buffers are not permitted, so just use a dummy in that case.
            // \todo clean this up; preferably stuff it into a w/ seeds and w/o seeds pass?
            auto const buf_open_vals    = open_list_vals.empty() ? cl::Buffer() : cl::Buffer(info.ctx, open_list_vals .begin(), open_list_vals .end(), true, true);
            auto const buf_open_sizes   = open_list_vals.empty() ? cl::Buffer() : cl::Buffer(info.ctx, open_list_sizes.begin(), open_list_sizes.end(), true, true);
            auto const&buf_open         = open_list_vals.empty() ? buffer_empty : buf_open_vals;
            auto const&buf_open_scan    = open_list_vals.empty() ? buffer_empty : buf_open_sizes;
            
            cl::Event event_buf_jobs_ready;
            info.cmd_queue.enqueueWriteBuffer(buf_jobs, CL_FALSE, 0
                                             ,num_jobs * sizeof(job_ids[0]), job_ids.data()
                                             ,nullptr, &event_buf_jobs_ready);

            auto const global_sz  = cl::NDRange(wg_size_max * num_jobs);
            auto const workgrp_sz = cl::NDRange(wg_size_max);
            auto& k = do_debug ? kernel_dbg : kernel;
            auto const job_event  = k(cl::EnqueueArgs(info.cmd_queue, event_buf_jobs_ready, global_sz, workgrp_sz)
                                          ,info.config.fim_minimum_change
                                          ,buf_cost   , buf_time, to_cl_ui2(field_cost.size())
                                          ,buf_status , buf_jobs
                                          ,!open_list_vals.empty(), buf_open, buf_open_scan
                                          );

            cl::vector<cl::Event> await_events {job_event};
            // turns out its less expensive on AMD to do a single huge read than try and be fancy
            // about picking out exactly which subset we want.
            // (no coalescing or API overhead? I hope its just API overhead)
#if 1
            auto const status_read_offset = job_id_min;
            auto const status_read_length = (job_id_max - job_id_min) + 1;
            info.cmd_queue.enqueueReadBuffer(buf_status, CL_TRUE, 0
                                            ,num_jobs * sizeof(BlockStatus)
                                            ,job_result.data()
                                            ,&await_events);
#else
            for (size_t i = 0; i < num_jobs; ++i) {
                auto const& job = job_ids[i];
                cmdQueue.enqueueReadBuffer(buf_status, CL_FALSE, job.id * sizeof(BlockStatus)
                                          ,sizeof(BlockStatus)
                                          ,block_infos.status.data() + job.id
                                          ,&await_events);
            }
            cl::Event event_all_jobs;
            cmdQueue.enqueueMarkerWithWaitList(nullptr, &event_all_jobs);
            event_all_jobs.wait();
#endif

            for (size_t i = 0; i < num_jobs; ++i) {
                auto&       job       = job_ids[i];
                auto&       status    = block_infos.status[job.id];
                auto const  block_xy  = dim_pos(blocks, job.id);

                status = job_result[i];

                for (size_t i = 0; i < length_of(k_border_directions); ++i) {
                    // ensure we're boolean...
                    assert((status.border[i] == 1)
                        || (status.border[i] == 0));

                    auto const pos   = int2(block_xy) + k_border_directions[i];
                    auto const dirty = status.border[i] && dim_in_bounds(int2(blocks), pos);
                    if (!dirty) continue;

                    
                    auto const idx = (BlockId)dim_idx(blocks, uint2(pos));
                    if (!blocks_dirty_set[idx]) {
                        blocks_dirty_set[idx] = true;
                        blocks_dirty_queue.push(BlockWrapper { &block_infos, idx  });
                    }

                    auto const border_id = (i == 0) ? 2 :
                                           (i == 1) ? 3 :
                                           (i == 2) ? 0 :
                                                      1;
                    //std::cout << "Dirty Border: " << block_xy.x << ", " << block_xy.y << " -> " << pos.x << ", " << pos.y << "\n\tborder: " << i << " -> " << border_id << "\n";
                    blocks_border_mask[idx] |= 1 << (8 * border_id);
                }

                std::fill(status.border, status.border + length_of(status.border), 0);
            }

            for (size_t i = 0; i < num_jobs; ++i)
                blocks_dirty_set[job_ids[i].id] = false;

            auto const device_time_ns = job_event.getProfilingInfo<CL_PROFILING_COMMAND_END   >()
                                      - job_event.getProfilingInfo<CL_PROFILING_COMMAND_START >();
            for (size_t i = 0; i < num_jobs; ++i) {
                auto const& job = job_ids[i];
                ProfilingInfo info {};
                info.block_id       = job.id;
                info.device_time_ns = device_time_ns / num_jobs;
                info.total_improved = block_infos.status[job.id].cells_updated;
                info.total_solves   = block_infos.status[job.id].solves;
                profiling_info.push_back(info);
            }
        }

        //assert(profiling_info.size() >= explored_blocks.size());
        assert(seed_map.empty());
        info.cmd_queue.enqueueReadBuffer(buf_time, CL_TRUE, 0
                                        ,field_time.size_flat() * sizeof(float)
                                        ,field_time.backing_store().data());

        auto const whole_end    = std::chrono::high_resolution_clock::now();

        size_t total_device_time_ns = 0; // in nano-second
        size_t total_solves         = 0;
        size_t total_cell_improves  = 0;
        for (auto&& i : profiling_info) {
            total_device_time_ns += i.device_time_ns;
            total_solves         += i.total_solves;
            total_cell_improves  += i.total_improved;
        }

        result.duration_total   = std::chrono::duration_cast<EikonalResult::duration_t>(whole_end - whole_bgn);
        result.duration_device  = std::chrono::duration_cast<EikonalResult::duration_t>(std::chrono::nanoseconds(total_device_time_ns));

        auto const per_blk_device_time_ns = total_device_time_ns / profiling_info.size();
        auto const per_blk_solves         = total_solves         / profiling_info.size();
        auto const per_blk_cell_improves  = total_cell_improves  / profiling_info.size();
        auto const per_cell_solves        = double(total_solves       ) / field_time.size_flat();
        auto const per_cell_improves      = double(total_cell_improves) / field_time.size_flat();
        std::cout << "GPU HCM Execution Complete.\n"
                  << info.config
                  << "\tEntire Time: "; printf("%0.3f (ms)\n", result.duration_total.count() / 1e3f);
        std::cout << "\tDevice Time: "; printf("%0.3f (ms)\n", total_device_time_ns / 1e6f);
        std::cout << "\t\tAvg Block: " << per_blk_device_time_ns << " (ns)\n";
        std::cout << "\t\tClock Res: " << info.device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() << " (ns)\n"
                  << "\tTotal:\n"
                  << "\t\t# of solves:       " << total_solves << "\n"
                  << "\t\t# of updates:      " << total_cell_improves << "\n"
                  << "\tPer Block:\n"
                  << "\t\tAvg # of enqueues: " << profiling_info.size() / double(blocks.x * blocks.y) << "\n"
                  << "\t\tAvg # of solves:   " << per_blk_solves  << "\n"
                  << "\t\tAvg # of updates:  " << per_blk_cell_improves << "\n"
                  << "\tPer Cell:\n"
                  << "\t\tAvg # of solves:   " << per_cell_solves << "\n"
                  << "\t\tAvg # of updates:  " << per_cell_improves << "\n";
    });
    std::cout << "Num Iter: " << num_wave << "\n";
    result.field = std::move(field_time);
    return result;
}

