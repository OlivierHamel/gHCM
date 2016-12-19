#pragma once


struct OclDevInfo {
    cl::Device  device;
    std::string name, vendor, version, profile, type;
    std::vector<std::string> extensions;

    OclDevInfo() = default;
    OclDevInfo(OclDevInfo const&) = default;
    OclDevInfo(cl::Device);

    size_t    maxWorkGroup, maxComputeUnits, maxConstantBuffer;
    cl_uint   hzMax;
    cl_ulong  memLocal, memGlobal;
    cl_bool   available;

    bool operator<(OclDevInfo const& b) {
        return std::tie(  type,   available,   hzMax,   memGlobal,   memLocal
                       ,  maxComputeUnits,   maxWorkGroup,   maxConstantBuffer) <
               std::tie(b.type, b.available, b.hzMax, b.memGlobal, b.memLocal
                       ,b.maxComputeUnits, b.maxWorkGroup, b.maxConstantBuffer);
    }
};

std::ostream& operator<<(std::ostream& os, OclDevInfo const& di);

std::vector<OclDevInfo>   cl_platform_devices(cl::Platform const&, cl_device_type);
void                      cl_platform_device_dump(cl::Platform const&, cl_device_type
                                                 ,std::ostream&);

optional<OclDevInfo>      cl_device_select_first(std::vector<OclDevInfo> const& devices
                                                ,cl_device_type                 type);

std::vector<std::string>  cl_splice_extensions(char const*);

std::string               cl_device_type_string(cl_device_type);
optional<cl_device_type>  cl_device_type_parse (char const*);

optional<cl::Program>     cl_util_build(cl::Context const&
                                       ,cl::vector<cl::Device> const&
                                       ,char const* build_options
                                       ,char const* file_path_no_ext);
const char* cl_error_message(cl_int error);

template<typename F>
auto cl_do(char const* const taskName, F&& task) -> std::result_of_t<F()> {
    try { return task(); }
    catch (const cl::Error& e) {
        auto const who = e.what() ? e.what() : "<null>";
        auto const err = cl_error_message(e.err());
        std::cout << "OpenCL Error.\n\tTask: " << taskName << "\n\tWho: " << who << "\n\tErr: " << err << "\n";
        getchar();
        exit(-1);
    }
}

