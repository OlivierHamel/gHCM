
#include "test_runner.h"
#include "eikonal.h"
#include "util_field.h"
#include "util_files.h"
#include "field_loaders.h"


namespace {

auto const k_fim_minimum_change = 1e-3f;
auto const k_num_trials         = 5;

template<typename F /* char const -> optional<Field2D<float>> */>
Field2D<float> field_load_checked(char const* const file_path, F&& f) {
    auto field = f(file_path);
    if (!field) {
        std::cout << "Failed to load img file: " << file_path << "\n";
        exit(-1);
        return{ };
    }

    return std::move(*field);
}

Field2D<float> field_speed_from_img(char const* const file_path)
{ return field_load_checked(file_path, field2d_img_monochannel); }

Field2D<float> field_speed_from_img_luminosity(char const* const file_path)
{ return field_load_checked(file_path, field2d_img_rgb_luminosity); }

Field2D<float> field_cost_from_speed(Field2D<float> f) {
    for (auto& v : f.backing_store()) v = 1 / v;

    return f;
}

Field2D<float> field_cost_from_img(char const* const file_path)
{ return field_cost_from_speed(field_speed_from_img(file_path)); }

Field2D<float> field_cost_constant(uvec2 const sz)
{ return Field2D<float>(sz, 1.f); }

template<typename F /* vec2 \in [0, 1] -> float */>
Field2D<float> field_cost_parameteric(uvec2 const sz, F&& f) {
    auto field = Field2D<float>(sz);
    for (size_t i = 0; i < field.size_flat(); ++i)
        field[i] = f(vec2(dim_pos(field.size(), i)) / vec2(sz));// -vec2(.5f));

    return field;
}

Field2D<float> field_cost_sin_checkerboard(uvec2 const sz, float const speed_amplitude
                                                         , float const checkerboard_res) {
    return field_cost_parameteric(sz, [&](vec2 const xy) {
        auto const v = sin(float(checkerboard_res * 2 * M_PI) * xy);
        return float(1 + speed_amplitude * v.x * v.y);
    });
}

Field2D<float> field_cost_random(uvec2 const sz, size_t seed) {
    using rnd_t   = std::default_random_engine;
    auto  rnd_gen = rnd_t(seed);

    return field_cost_parameteric(sz, [&](vec2) {
        auto const n = float(rnd_gen() - rnd_t::min()) / rnd_t::max();
        return 1 + n * 5;
    });
}

Field2D<float> field_cost_as_permeable(Field2D<float> f, float const inf_replacement = 1e3) {
    for (auto&& x : f.backing_store())
        x = std::min(x, inf_replacement);

    return f;
}


template<typename T>
struct BenchmarkAsyncResult {
    std::chrono::high_resolution_clock::time_point time_bgn, time_end;
    T return_value;
};

template<typename F, typename T = std::result_of_t<F()>>
std::future<BenchmarkAsyncResult<T>>
task_async(char const* name, F&& f) {
    std::cout << "Benchmark CPU Task (async) - Enqueue: " << name << "\n";
    return std::async(std::launch::async, [](std::function<T()> f) {
        BenchmarkAsyncResult<T> result;
        result.time_bgn     = std::chrono::high_resolution_clock::now();
        result.return_value = f();
        result.time_end     = std::chrono::high_resolution_clock::now();

        return std::move(result);
    }, std::function<T()>(f));
}

template<typename T>
T task_async_get(char const* name, std::future<BenchmarkAsyncResult<T>> f) {
    auto       result = f.get();
    auto const diff   = result.time_end - result.time_bgn;
    auto const t      = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

    std::cout << "Benchmark CPU Task (async) - Result:\n\tName: " << name << "\n";
    printf("\tDuration: %0.3f (ms)\n", t / 1e3f);

    return std::move(result.return_value);
}


struct Job {
    std::string     name;
    Field2D<float>  field;
};

using SolverFn = std::function<EikonalResult(Field2D<float> const&, std::vector<uint2> const&)>;

std::string hcm_config_name(HcmConfig const& config) {
    auto const kernel_mode = ([&] {
        switch (config.kernel) {
            case HcmConfig::Kernel::Systolic: return "systolic";
            case HcmConfig::Kernel::FIM: return "fim";
        }
        assert(false);
        return "unknown";
    })();

    auto const cache_mode = ([&] {
        switch (config.cache_mode) {
            case HcmConfig::CacheMode::None: return "cache_none";
            case HcmConfig::CacheMode::Time: return "cache_time";
            case HcmConfig::CacheMode::Cost: return "cache_cost";
            case HcmConfig::CacheMode::Both: return "cache_both";
        }
        assert(false);
        return "unknown";
    })();

    std::stringstream name;
    name << "ghcm_" << kernel_mode << "_" << cache_mode << "_cu_over_" << size_t(config.compute_unit_overload) << "_wg_over_" << size_t(config.workers_overload) << "_block_sz_" << size_t(config.block_size);
    return name.str();
};

auto const field_diff = [](Field2D<float> const& a
                          ,Field2D<float> const& b) {
    assert(a.size() == b.size());
    // b/c having a std::zip(a, b, c, ...) would be too much to bloody ask for...
    Field2D<float> diff(a.size(), INFINITY);
    for (size_t i = 0; i < diff.size_flat(); ++i) {
        auto const both_inf = isinf(a.backing_store()[i]) && isinf(b.backing_store()[i]);
        diff.backing_store()[i] = both_inf ? INFINITY : eikonal_time_diff(a.backing_store()[i], b.backing_store()[i]);
    }

    return diff;
};

}

void test_run(std::string         const& solver_name
             ,std::vector<Job   > const& jobs
             ,std::vector<uint2 > const& seeds
             ,SolverFn            const& fn) {
    auto const base_path = "data/trials/" + solver_name;

    for (auto&& job : jobs) {
        auto const csv_file_path = base_path + "_" + job.name + ".csv";
        if (auto const f  = fopen(csv_file_path.c_str(), "rb")) {
            auto const sz = file_size(f);
            fclose(f);
            if (sz && *sz) {
                std::cout << "Skipping: " << solver_name << " | " << job.name << "\n";
                continue;
            }
        }

        std::stringstream result_csv;

        EikonalResult::duration_t dur_device[k_num_trials];
        EikonalResult::duration_t dur_total [k_num_trials];
        EikonalResult result;
        for (size_t i = 0; i < k_num_trials; ++i) {
            std::cout << "Running: " << solver_name << ", " << job.name << ", " << i << " of " << k_num_trials << "\n";
            result = fn(job.field, seeds);
            dur_device[i] = result.duration_device;
            dur_total [i] = result.duration_total;
        }

        std::stringstream path_blob; path_blob << base_path << "_" << job.name << ".blob";
        size_t const x = result.field.size().x, y = result.field.size().y;
        auto const f = fopen(path_blob.str().c_str(), "wb"); assert(f);
        fwrite(&x, sizeof(x), 1, f);
        fwrite(&y, sizeof(y), 1, f);
        fwrite(result.field.backing_store().data(), sizeof(float), result.field.backing_store().size(), f);
        fclose(f);

        file_write_time_field((base_path + "_" + job.name + ".png").c_str(), result.field);
        result_csv << solver_name << "," << job.name << ", device";
        for (auto&& d : dur_device) result_csv << "," << d.count();
        result_csv << "\n";

        result_csv << solver_name << "," << job.name << ", total";
        for (auto&& d : dur_total) result_csv << "," << d.count();
        result_csv << "\n";

        auto csv_stream = std::ofstream(base_path + "_" + job.name + ".csv", std::ofstream::out);
        csv_stream << result_csv.str();
        csv_stream.close();
    }

    
}


void test_ghcm(cl::Device          const& device
              ,std::vector<Job   > const& jobs
              ,std::vector<uint2 > const& seeds) {
    

    auto const run_config = [&](HcmConfig const& config_base, bool const test_various_worker_overload) {
        u32 const worker_overloads[]   = { 1, 2, 4 };
        u32 const compute_overloads[]  = { 1, 2, 4, 8, 16, 32 };
        u8  const block_sizes[]        = { 8, 16, 32 };
        HcmConfig::CacheMode cache_modes[] = { HcmConfig::CacheMode::Both
                                             // literally do not have time to wait for these to finish chewing
                                             /*, HcmConfig::CacheMode::Time
                                             , HcmConfig::CacheMode::Cost
                                             , HcmConfig::CacheMode::None */ };

        auto   const subvariants        = [&](HcmConfig& c) {
            for (auto&& cache_mode : cache_modes) {
                for (auto&& compute_overload : compute_overloads) {
                    for (auto&& block_size : block_sizes) {
                        c.compute_unit_overload = compute_overload;
                        c.block_size            = block_size;
                        c.cache_mode            = cache_mode;

                        if (auto info = eikonal_hcm_opencl_setup(device, c))
                            test_run(hcm_config_name(c), jobs, seeds
                                    ,std::bind(eikonal_hcm_opencl_ex, *info
                                              ,std::placeholders::_1, std::placeholders::_2));
                        else
                            std::cerr << "\n\nError: Failed setup for: " << hcm_config_name(c) << "\n\n\n";
                    }
                }
            }
        };

        if (test_various_worker_overload) {
            for (auto&& worker_overload : worker_overloads) {
                auto config = config_base;
                config.workers_overload = worker_overload;
                subvariants(config);
            }
        } else {
            auto config = config_base;
            config.workers_overload = 1;
            subvariants(config);
        }
    };

    HcmConfig config_base {};
    config_base.fim_minimum_change = k_fim_minimum_change;

    // systolic
    run_config(([&] {
        auto c = config_base;
        c.kernel                = HcmConfig::Kernel::Systolic;
        c.workers_overload      = 1;
        return c;
    })(), false);

    // FIM
    run_config(([&] {
        auto c = config_base;
        c.kernel                = HcmConfig::Kernel::FIM;
        c.workers_overload      = 1;
        return c;
    })(), true);
}


void tests_all(cl::Platform const&, cl::Device const& device) {
#if 1
    HcmConfig hcm_config_base {};
    hcm_config_base.fim_minimum_change      = k_fim_minimum_change;
    hcm_config_base.cache_mode              = HcmConfig::CacheMode::Both;
    hcm_config_base.kernel                  = HcmConfig::Kernel::FIM;
    hcm_config_base.compute_unit_overload   = 10; // high enough and we're basically Block FIM
    hcm_config_base.workers_overload        = 1;
    hcm_config_base.block_size              = 8;
    auto const field = field_speed_from_img_luminosity("data/twin_foals.jpeg");
    file_write_time_field("data/cost_field.png", field);

    auto const output = eikonal_hcm_opencl(device, hcm_config_base, field, {uint2(0)}).field;
    file_write_field_hdr("data/tmp.hdr", output);
    file_write_time_field("data/tmp.png", output);
#else
    auto const problem_7ring    = field_cost_from_img("data/7_ring_4096_2d.png");
    auto const problem_cloudy   = field_cost_from_img("data/cloudy_4096_2d.png");
    auto const problem_equine   = field_speed_from_img("data/denormal_scale_equine.png");
    auto const parameteric_sz   = uvec2(1u << 12);
    auto const seed_points      = std::vector<uint2> { uint2(0) };//field_cost.size() / uint2(2) };
    std::vector<Job> jobs
        { { "7 rings"           , problem_7ring                             }
        , { "7 rings permeable" , field_cost_as_permeable(problem_7ring)    }
        , { "cloudy"            , problem_cloudy                            }
        , { "photograph"        , problem_equine                            }
        , { "constant"          , field_cost_constant(parameteric_sz)       }
        , { "sin checkerboard"  , field_cost_sin_checkerboard(parameteric_sz, .75, 20) }
        };

    auto const handle_fmm = std::async(std::launch::async, test_run, "fmm", jobs, seed_points, eikonal_fmm_sequential);
    auto const handle_fim = std::async(std::launch::async, test_run, "fim", jobs, seed_points, std::bind(eikonal_fim_sequential, k_fim_minimum_change, std::placeholders::_1, std::placeholders::_2));
    test_ghcm(device, jobs, seed_points);

    handle_fmm.wait();
    handle_fim.wait();

    for (auto&& j : jobs) {
        HcmConfig config_base {};
        config_base.fim_minimum_change = k_fim_minimum_change;
        config_base.kernel = HcmConfig::Kernel::Systolic;

        auto const load_blob = [&](std::string const& s) {
            auto const f = fopen(s.c_str(), "rb"); assert(f);
            size_t x, y;
            fread(&x, sizeof(x), 1, f);
            fread(&y, sizeof(y), 1, f);
            Field2D<float> field(uvec2(x, y));
            fread(field.backing_store().data(), sizeof(float), field.backing_store().size(), f);
            fclose(f);
            return field;
        };

        std::stringstream path_blob_fmm;  path_blob_fmm  << "data/trials/fmm_" << j.name << ".blob";
        std::stringstream path_blob_ghcm; path_blob_ghcm << "data/trials/" << config_name(config_base) << "_" << j.name << ".blob";
        std::stringstream ss; ss << "data/trails_fmm_ghcm_diff_" << j.name << ".png";
        file_write_time_field(ss.str().c_str(), field_diff(load_blob(path_blob_ghcm.str())
                                                          ,load_blob(path_blob_fmm.str())));
    }
#endif
}


