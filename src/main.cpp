
#include "eikonal.h"
#include "test_runner.h"
#include "util_field.h"
#include "util_files.h"
#include "util_opencl.h"

namespace {

#ifdef _WIN32
// http://stackoverflow.com/questions/5248704/how-to-redirect-stdout-to-output-window-from-visual-studio
/// \brief This class is a derivate of basic_stringbuf which will output all the written data using the OutputDebugString function
template<typename TChar>
class OutputDebugStringBuf : public std::basic_stringbuf<TChar, std::char_traits<TChar>> {
public:
    using TTraits = std::char_traits<TChar>;
    static_assert(std::is_same<TChar,char>::value || std::is_same<TChar,wchar_t>::value
                 ,"OutputDebugStringBuf only supports char and wchar_t types");

    explicit OutputDebugStringBuf() : _buffer(256) {
        setg(nullptr, nullptr, nullptr);
        setp(_buffer.data(), _buffer.data(), _buffer.data() + _buffer.size());
    }

    int sync() {
        try {
            MessageOutputer<TChar,TTraits>()(pbase(), pptr());
            setp(_buffer.data(), _buffer.data(), _buffer.data() + _buffer.size());
            return 0;
        } catch(...) {
            return -1;
        }
    }

    int overflow(int c = TTraits::eof()) {
        auto syncRet = sync();
        if (c != TTraits::eof()) {
            _buffer[0] = char(c);
            setp(_buffer.data(), _buffer.data() + 1, _buffer.data() + _buffer.size());
        }
        return syncRet == -1 ? TTraits::eof() : 0;
    }


private:
    std::vector<TChar>      _buffer;

    template<typename TChar, typename TTraits>
    struct MessageOutputer;

    template<>
    struct MessageOutputer<char,std::char_traits<char>> {
        template<typename TIterator>
        void operator()(TIterator begin, TIterator end) const {
            std::string s(begin, end);
            OutputDebugStringA(s.c_str());
        }
    };

    template<>
    struct MessageOutputer<wchar_t,std::char_traits<wchar_t>> {
        template<typename TIterator>
        void operator()(TIterator begin, TIterator end) const {
            std::wstring s(begin, end);
            OutputDebugStringW(s.c_str());
        }
    };
};

static OutputDebugStringBuf<char    > charDebugOutput;
static OutputDebugStringBuf<wchar_t > wcharDebugOutput;
#endif


auto const kDocOptsUsage = R"[[(
comp5704project
    olivier hamel - comp5704 project - eikonal gpu solver

Usage:
    comp5704project [options]
    comp5704project info [options]
    comp5704project (-h | --help)
    comp5704project --version

info                Prints info regarding the default OpenCL platform on this system.

Options:
    -h --help               Show this screen.
    --version               Show version.
    -d --devicetype <type>  Specify which device type to use. Options: gpu, cpu, any [default: gpu]
)[[";

}


int main(int const argc, char const* const argv[]) {
#ifdef _WIN32
    if (IsDebuggerPresent()) {
        std::cerr.rdbuf(&charDebugOutput);
        std::clog.rdbuf(&charDebugOutput);
        //std::cout.rdbuf(&charDebugOutput);

        std::wcerr.rdbuf(&wcharDebugOutput);
        std::wclog.rdbuf(&wcharDebugOutput);
        //std::wcout.rdbuf(&wcharDebugOutput);
    }
#endif

    auto const cmd_args     = docopt::docopt(kDocOptsUsage, { argv + 1, argv + argc }
                                            ,true                   /* show help if requested */
                                            ,"comp5704 project 0.1" /* version string */);
    auto const device_type  = cl_device_type_parse(cmd_args.at("--devicetype").asString().c_str());
    if (!device_type) {
        std::cout << "Invalid --devicetype value: " << cmd_args.at("--devicetype").asString() << "\n"
                  << "    See --help for valid device types.\n";
        return -1;
    }

    auto const platform     = cl_do("getting default platform object"
                                   ,[] { return cl::Platform::getDefault(); });
    if (cmd_args.at("info").asBool()) {
        cl_do("dumping platform info"
             ,[&] { cl_platform_device_dump(platform, *device_type, std::cout); getchar(); });
        return 0;
    }


    auto const device_info  = cl_do("fetching device", [&] {
        return cl_device_select_first(cl_platform_devices(platform, *device_type), *device_type);
    });
    if (!device_info) {
        std::cout << "No device satisfying requirements found. Requirements:\n"
                  << "  Device Type: " << cl_device_type_string(*device_type) << "\n";
        return -1;
    }
    std::cout << "Selected " << *device_info;


    
    tests_all(platform, device_info->device);





    
#if 0
    auto       field_cost           = field_cost_from_img("data/7_ring_4096_2d.png");
    field_cost_mutate_to_permeable(field_cost);
    //auto const field_cost           = field_cost_constant(uvec2(1 << 14));
    //auto const field_cost           = field_cost_random(uvec2(1 << 12), 1);
    //auto const field_cost = field_cost_sin_checkerboard(uvec2(1 << 12), .5, 20);
    /*auto const field_cost           = field_cost_parameteric(uvec2(1 << 12), [&](vec2 const xy) {
        auto const speed_amplitude = .5;
        auto const v = sin(10 * 2 * M_PI * xy + max(sin(vec2(xy.x, 0) * M_PI * 2.f * 10.f), vec2(0)) * M_PI * 2.f);
        return float(1 + speed_amplitude * v.x * v.y);
    });*/
    std::cout << "Field Size: " << field_cost.size().x << ", " << field_cost.size().y << "\n";

    double sum = 0, sum_sq = 0;
    for (auto&& v : field_cost.backing_store()) {
        sum += v;
        sum_sq += v * v;
    }
    auto const sum_avg = sum / field_cost.size_flat();
    double std_dv = sqrt(sum_sq / field_cost.size_flat() - sum_avg * sum_avg);
    std::cout << "Standard speed deviation: " << std_dv << "\n"
              << "Avg Speed: " << sum_avg << "\n";

    auto const seed_points          = std::vector<uint2> { uint2(0) };//field_cost.size() / uint2(2) };
    /*auto       fmm_solution_handle  = task_async("FMM baseline",
        [&] { return eikonal_fmm_sequential(                      field_cost, seed_points); });*/
    /*auto       fim_solution_handle  = task_async("FIM baseline",
        [&] { return eikonal_fim_sequential(k_fim_minimum_change, field_cost, seed_points); });*/

    HcmConfig hcm_config_base {};
    hcm_config_base.fim_minimum_change      = k_fim_minimum_change;
    hcm_config_base.cache_mode              = HcmConfig::CacheMode::Both;
    hcm_config_base.compute_unit_overload   = 4; // high enough and we're basically Block FIM
    hcm_config_base.workers_overload        = 1;
    hcm_config_base.block_size              = 8;
    //hcm_config_base.cache_mode              = HcmConfig::CacheMode::Both;
    //hcm_config_base.compute_unit_overload   = 10; // high enough and we're basically Block FIM
    //hcm_config_base.workers_overload        = 4;
    //hcm_config_base.block_size              = 16;

    auto       hcm_solution = eikonal_hcm_opencl(device_info->device, hcm_config_base
                                                ,field_cost, seed_points);
    //auto const fmm_solution = task_async_get("Baseline FMM", std::move(fmm_solution_handle));
    //auto const fim_solution = task_async_get("Baseline FIM", std::move(fim_solution_handle));

    

    

    //file_write_time_field("data/cost_field.png", field_cost);

    //file_write_time_field("data/solution_fmm.png", fmm_solution);
    //file_write_time_field("data/solution_fim.png", fim_solution);
    file_write_time_field(("data/solution_" + config_name(hcm_config_base) + ".png").c_str()
                         ,hcm_solution);
    //write_diff("fim_vs_fmm", fim_solution, fmm_solution);
    //write_diff((config_name(hcm_config_base) + "_vs_fmm").c_str(), hcm_solution, fmm_solution);
    //write_diff((config_name(hcm_config_base) + "_vs_fim").c_str(), hcm_solution, fim_solution);

    auto const test_alternate_config = [&](HcmConfig::CacheMode::Type const cache_mode
                                          ,u8                         const block_size) {
        HcmConfig hcm_config_extra  = hcm_config_base;
        hcm_config_extra.cache_mode = cache_mode;
        hcm_config_extra.block_size = block_size;

        auto solution = eikonal_hcm_opencl(device_info->device, hcm_config_extra
                                          ,field_cost, seed_points);
        file_write_time_field(("data/solution_" + config_name(hcm_config_extra) + ".png").c_str()
                             ,solution);
        write_diff((config_name(hcm_config_extra) + "_vs_" + config_name(hcm_config_base)).c_str()
                  ,solution, hcm_solution);
    };

    for (auto cache_mode = HcmConfig::CacheMode::None;
              cache_mode < HcmConfig::CacheMode::COUNT__;
              cache_mode = HcmConfig::CacheMode::Type(cache_mode + 1)) {
        for (u8 block_size = 8; block_size <= 32; block_size *= 2)
            ;// test_alternate_config(cache_mode, block_size);
    }
#endif
    std::cout << "Now what?\n";
    getchar();
    return 0;
}

