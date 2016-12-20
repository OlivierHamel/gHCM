
#include "util_files.h"


namespace {

// shamelessly butchered from wikipedia. tis only for (half-assed) visualisation.
std::tuple<unsigned char, unsigned char, unsigned char> hue_to_rgb(float const h /*\in [0, 1]*/) {
    auto const s = std::fmod(h, 1) * 6;
    auto const x = (unsigned char)((1 - std::abs(std::fmod(s, 2) - 1)) * 255);
    switch (int(s)) {
    case 0: return { 255,   x,  0  };
    case 1: return {   x, 255,  0  };
    case 2: return {  0 , 255,   x };
    case 3: return {  0 ,   x, 255 };
    case 4: return {   x,  0 , 255 };
    case 5: return { 255,  0 ,   x };
    }
    assert(false && "unreachable");
    return { 0, 0, 0 };
};

}


optional<size_t> file_size(FILE* const f) {
    assert(f);
    if (!f                        ) return {};
    if (fseek(f, 0, SEEK_END) != 0) return {};

    auto const len = ftell(f);
    if (len == -1                 ) return {};
    assert(0 <= len);
    return len;
}

bool file_write_field_hdr(char const*    const  file_name
                         ,Field2D<float> const& field_time) {
    auto const err = stbi_write_hdr(file_name, field_time.size().x, field_time.size().y, STBI_grey
                                   ,field_time.backing_store().data());
    if (err == 0) {
        std::cerr << "Failed to write hdr output.\n";
        return false;
    }

    return true;
}

bool file_write_time_field(char const*    const  file_name
                          ,Field2D<float> const& field_time) {
    float f_min = INFINITY, f_max = -INFINITY;
    for (auto&& x : field_time.backing_store())
        if (!isinf(x)) { f_min = std::min(f_min, x); f_max = std::max(f_max, x); }

    return file_write_time_field(file_name, field_time, f_min, f_max);
}

bool file_write_time_field(char const*    const  file_name
                          ,Field2D<float> const& field_time
                          ,float          const  f_min
                          ,float          const  f_max) {
    std::cout   << "File: "  << file_name   << "\n"
                << "\tMin: " << f_min       << "\n"
                << "\tMax: " << f_max       << "\n";

    auto const hued_pixels = ([&] {
        std::vector<unsigned char> hue(field_time.size_flat() * 3, 0);
        if (f_max <  f_min) return hue; // display range is null

        // if range is singleton, then just give it a uniform colour.
        auto const range  = (f_min == f_max) ? 1 : (f_max - f_min);
        auto const to_hue = [&](float x) { return (5 / 6.f) * (x - f_min) / range; };
        auto       out    = hue.data();
        for (auto&& x : field_time.backing_store()) {
            if (!isinf(x))
                std::tie(out[0], out[1], out[2]) = hue_to_rgb(to_hue(x));

            out += 3;
        }

        return hue;
    })();

    auto const err = stbi_write_png(file_name, field_time.size().x, field_time.size().y, STBI_rgb
                                   ,hued_pixels.data(), hued_pixels.size() / field_time.size().y);
    if (err == 0) {
        std::cerr << "Failed to write output.\n";
        return false;
    }

    return true;
}