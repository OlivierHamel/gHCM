#pragma once

#include "util_field.h"

// direct load of an mono-channel (grayscale) image
optional<Field2D<float>> field2d_img_monochannel(char const* file_path);

// converts rgb img to grayscale (human perceptional luminosity)
// \note does not consider colour-space metadata, unless STB has changed recently to handle that
optional<Field2D<float>> field2d_img_rgb_luminosity(char const* file_path);
