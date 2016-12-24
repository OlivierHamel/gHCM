
#include "util_files.h"
#include "util_opencl.h"



namespace {
std::vector<std::string> split_string_by_spaces_dump_empty(char const* const s) {
    if (!(s && *s)) return {};

    std::vector<std::string> exts;
    auto const addExt = [&](char const* const head, char const* const end) {
        auto const len = end - head;
        if (0 < len) exts.push_back(std::string(head, len));
    };

    auto const  s_end = s + strlen(s);
    char const* head  = s;
    while (char const* end = strchr(head, ' ')) {
        addExt(head, end);
        head = end + 1;
    }

    // last one isn't picked up by the main loop
    addExt(head, s + strlen(s));
    return exts;
}

std::string display_format_mem(cl_ulong const bytes) {
    char const* k_suffixes[] = {"bytes", "KiB", "MiB", "GiB"};
    auto const  k_num_suffix = sizeof(k_suffixes) / sizeof(k_suffixes[0]);

    size_t scale  = 0;
    double f      = (double)bytes;
    for (; (scale < k_num_suffix) && (1024 < f); ++scale)
        f /= 1024;

    char buffer[512];
    snprintf(buffer, length_of(buffer), "%f %s", f, k_suffixes[scale]);
    return buffer;
}

}

OclDevInfo::OclDevInfo(cl::Device d) {
    device              = d;
    name                = d.getInfo<CL_DEVICE_NAME>();
    vendor              = d.getInfo<CL_DEVICE_VENDOR>();
    version             = d.getInfo<CL_DEVICE_VERSION>();
    profile             = d.getInfo<CL_DEVICE_PROFILE>();
    type                = cl_device_type_string(d.getInfo<CL_DEVICE_TYPE>());
    hzMax               = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    memLocal            = d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    memGlobal           = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    available           = d.getInfo<CL_DEVICE_AVAILABLE>();
    maxWorkGroup        = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    maxComputeUnits     = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    maxConstantBuffer   = d.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
    extensions          = cl_splice_extensions(d.getInfo<CL_DEVICE_EXTENSIONS>().c_str());
}

std::vector<OclDevInfo> cl_platform_devices(cl::Platform const& p, cl_device_type const type) {
    cl::vector<cl::Device> devices; p.getDevices(type, &devices);

    std::vector<OclDevInfo> l(devices.size());
    std::transform(devices.begin(), devices.end(), l.begin()
                    , [](cl::Device& d) { return OclDevInfo(d); });
    std::sort(l.begin(), l.end());

    return l;
}

void cl_platform_device_dump(cl::Platform const& platform, cl_device_type const type
                            ,std::ostream& os) {
    auto extensions = cl_splice_extensions(platform.getInfo<CL_PLATFORM_EXTENSIONS>().c_str());
    std::sort(extensions.begin(), extensions.end());

    os  << "Platform: "       << platform.getInfo<CL_PLATFORM_NAME>()
        << "\n  Vendor    : " << platform.getInfo<CL_PLATFORM_VENDOR>()
        << "\n  Version   : " << platform.getInfo<CL_PLATFORM_VERSION>()
        << "\n  Profile   : " << platform.getInfo<CL_PLATFORM_PROFILE>()
        << "\n  Extensions:\n";

    for (auto&& s : extensions)
        os << "    " << s << "\n";

    auto const devices = cl_platform_devices(platform, type);
    os << "Devices reported (" << devices.size() << "):\n";
    for (auto&& d : devices) os << d;
}


optional<OclDevInfo> cl_device_select_first(std::vector<OclDevInfo> const& devices
                                           ,cl_device_type          const  type) {
    for (auto&& d : devices) {
        if (!d.available              ) continue;
        if (type == CL_DEVICE_TYPE_ALL) return d;

        auto const type = d.device.getInfo<CL_DEVICE_TYPE>();
        if ((type & type) == type     ) return d;
    }

    return {};
}


std::vector<std::string> cl_splice_extensions(char const* const s)
{ return split_string_by_spaces_dump_empty(s); }





std::string cl_device_type_string(cl_device_type const type) {
    if (type == 0) return "<none?>";

    std::string s;
    auto const add = [&](cl_device_type const flag, char const* name) {
        if (!(type & flag)  ) return;
        if (!s.empty()      ) s += " ";
        s += name;
    };
    add(CL_DEVICE_TYPE_CPU          , "CPU");
    add(CL_DEVICE_TYPE_GPU          , "GPU");
    add(CL_DEVICE_TYPE_ACCELERATOR  , "Accelerator");
    add(CL_DEVICE_TYPE_CUSTOM       , "Custom");
    return s.empty() ? "<unknown>" : s;
}

optional<cl_device_type> cl_device_type_parse(char const* const s) {
    auto const parts = split_string_by_spaces_dump_empty(s);
    if (parts.empty()) return {};

    cl_device_type  device_type = 0;
    for (auto&& p : parts) {
#if _WIN32
        auto const matches = [&](char const* const k) { return _strcmpi  (p.c_str(), k) == 0; };
#else
        auto const matches = [&](char const* const k) { return strcasecmp(p.c_str(), k) == 0; };
#endif

             if (matches("all"          )) return CL_DEVICE_TYPE_ALL;
        else if (matches("cpu"          )) device_type |= CL_DEVICE_TYPE_CPU;
        else if (matches("gpu"          )) device_type |= CL_DEVICE_TYPE_GPU;
        else if (matches("accelerator"  )) device_type |= CL_DEVICE_TYPE_ACCELERATOR;
        else if (matches("custom"       )) device_type |= CL_DEVICE_TYPE_CUSTOM;
        else    return {};
    }

    return device_type;
}

optional<cl::Program> cl_util_build(cl::Context const&            ctx
                                   ,cl::vector<cl::Device> const& devices
                                   ,char const* const build_options
                                   ,char const* const file_path_no_ext) {
    // \todo impl binary caching?
    auto const path_base    = std::string(file_path_no_ext);
    auto const path_src     = path_base + ".cl";
    auto       kernel_src   = file_read_all<char>(path_src.c_str());
    if (!kernel_src) {
        std::cerr << "Failed to read kernel file: " << path_src << "\n";
        return {};
    }
    kernel_src->push_back('\0'); // make it null-term as c-str

    auto prg = cl::Program(ctx, kernel_src->data(), false);
    try {
        prg.build(devices, build_options);
    } catch (cl::Error& e) {
        std::cerr << "Kernel Build Error: " << e.what() << "\n";

        for (auto&& d : devices) {
            std::cerr
                << "\tDevice: "           << d  .getInfo     <CL_DEVICE_NAME          >()  << "\n"
                << "\t\tBuild Status : "  << prg.getBuildInfo<CL_PROGRAM_BUILD_STATUS >(d) << "\n"
                << "\t\tBuild Options: "  << prg.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(d) << "\n"
                << "\t\tBuild Log    :\n" << prg.getBuildInfo<CL_PROGRAM_BUILD_LOG    >(d) << "\n";
        }

        throw e;
    }

    return prg;
}



std::ostream& operator<<(std::ostream& os, OclDevInfo const& d) {
    os  << "Device: "           << d.name
        << "\n  Available   : " << d.available
        << "\n  Type        : " << d.type
        << "\n  Vendor      : " << d.vendor
        << "\n  Version     : " << d.version
        << "\n  Profile     : " << d.profile
        << "\n  Clk Hz Max  : " << d.hzMax
        << "\n  Mem Local   : " << display_format_mem(d.memLocal  )
        << "\n  Mem Global  : " << display_format_mem(d.memGlobal )
        << "\n  workgroups Max      : " << d.maxWorkGroup
        << "\n  Compute Units Max   : " << d.maxComputeUnits
        << "\n  Constant Buffer Max : " << d.maxConstantBuffer
        << "\n  Extensions  :\n";

    for (auto&& s : d.extensions)
        os << "    " << s << "\n";

    return os;
}


// Could have linked CLEW, but that was too heavy weight.
// Could have src-parsed/munged my own, but someone's probably already done it.
//
// Kind soul who put this listing together:
//    http://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* cl_error_message(cl_int const error) {
    switch (error) {
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

