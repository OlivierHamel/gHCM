#pragma once

#include "util_field.h"


// POST-CONDITION: file position is undefined
optional<size_t> file_size(FILE* const f);

bool file_write_time_field(char const*    const  file_name
                          ,Field2D<float> const& field_time);

bool file_write_time_field(char const*    const  file_name
                          ,Field2D<float> const& field_time
                          ,float          const  f_min
                          ,float          const  f_max);

bool file_write_field_hdr (char const*    const  file_name
                          ,Field2D<float> const& field_time);


template<typename T>
optional<std::vector<T>> file_read_all(char const* const fileName) {
    assert(fileName && *fileName);
    if (!fileName ) return {};

    auto const f = fopen(fileName, "rb");
    if (!f        ) return {};

    auto const size = file_size(f);
    if (!size                           ) { fclose(f); return {}; }
    if (fseek(f, 0, SEEK_SET)      != 0 ) { fclose(f); return {}; }

    auto const num_elem_avail = *size / sizeof(T);
    std::vector<T> buf(num_elem_avail);
    assert(buf.size() == num_elem_avail);
    // \fixme handle partial/no read failure
    auto const num_elem = fread(buf.data(), sizeof(T), buf.size(), f);
    fclose(f);

    assert(buf.size() == num_elem);
    if (buf.size() != num_elem          )  return {};

    return std::move(buf);
}

