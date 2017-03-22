
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

    std::cout << "Run complete.\n";
    return 0;
}

