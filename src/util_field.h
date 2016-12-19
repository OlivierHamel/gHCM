#pragma once

#include "util_dim.h"


template<typename T>
class Field2D {
    std::vector<T>  _data;
    glm::u32vec2    _size { 0, 0 };

public:
    Field2D()               = default;
    Field2D(Field2D const&) = default;
    Field2D(Field2D&& m   ) { *this = std::move(m); }
    Field2D(glm::u32vec2 const sz                     ) : _size(sz)
    { _data.resize(size_flat()); }
    Field2D(glm::u32vec2 const sz, T const default_val) : _size(sz)
    { _data.resize(size_flat(), default_val); }

    Field2D(glm::u32vec2 const sz, T const values[/*sz.x * sz.y*/]) : _size(sz)
    { _data.resize(size_flat()); std::copy(values, values + size_flat(), _data.begin()); }

    Field2D& operator=(Field2D const&) = default;
    Field2D& operator=(Field2D&& m   ) {
        std::swap(_data, m._data);
        std::swap(_size, m._size);
        return *this;
    }

    size_t       size_flat() const { return dim_size(_size); }
    glm::u32vec2 size()      const { return _size; }

    bool in_bounds(glm::u32vec2 const& x) const
    { return dim_in_bounds(_size, x); }

    bool in_bounds(glm::i32vec2 const& x) const
    { return glm::all(glm::lessThanEqual(glm::i32vec2(0), x))
          && in_bounds(glm::u32vec2(x)); }

    T  get_or(T const& default_value, glm::i32vec2 const pos) const
    { return in_bounds(pos) ? (*this)[pos] : default_value; }


    std::vector<float>&       backing_store()       { return _data; }
    std::vector<float> const& backing_store() const { return _data; }

    T      & operator[](size_t const idx)       { assert(idx < size_flat()); return _data[idx]; }
    T const& operator[](size_t const idx) const { assert(idx < size_flat()); return _data[idx]; }

    T      & operator[](glm::u32vec2 const pos)       { return (*this)[dim_idx(_size, pos)]; }
    T const& operator[](glm::u32vec2 const pos) const { return (*this)[dim_idx(_size, pos)]; }

    T      & operator[](glm::i32vec2 const pos)
    { assert(in_bounds(pos)); return _data[dim_idx(_size, glm::u32vec2(pos))]; }
    T const& operator[](glm::i32vec2 const pos) const
    { assert(in_bounds(pos)); return _data[dim_idx(_size, glm::u32vec2(pos))]; }
};

