
#pragma once

// C++ headers
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

// C headers
#include <cinttypes>
#include <cstdio>

// OS headers
#if _WIN32
    #include <SDKDDKVer.h>
    #define NOMINMAX
    #include <windows.h>
#endif


// Libraries

// docopt
#include <docopt/docopt.h>

// OpenCL headers
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <clhpp/cl2.hpp>

// optional
#include <optional/optional.hpp>
using std::experimental::optional;

// STB
#include <stb/stb.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

// GLM
//#define GLM_MESSAGES
#define GLM_FORCE_EXPLICIT_CTOR
#define GLM_FORCE_SIZE_FUNC
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/constants.hpp>
#include <glm/glm/gtc/type_ptr.hpp>
#include <glm/glm/vector_relational.hpp>


// handy util
template<typename T, size_t K>
size_t length_of(T(&)[K]) { return K; }

