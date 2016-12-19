
template<typename T, size_t K>
size_t length_of(T(&)[K]) { return K; }

template<typename T, size_t K>
inline void sort_bubble(T (&ary)[K]) {
    for (uint j = 0; (j + 1) < K; ++j) {
        for (uint i = 0; (i + 1) < K; ++i) {
            T const a = ary[i     ];
            T const b = ary[i + 1 ];
            ary[i     ] = min(a, b);
            ary[i + 1 ] = max(a, b);
        }
    }
}

template<>
void sort_bubble<float, 2>(float (&ary)[2]) {
    float const a = ary[0];
    float const b = ary[1];
    ary[0] = min(a, b);
    ary[1] = max(a, b);
}

 int2   to_i2 ( char2 v) { return ( int2 )(v.x, v.y); }
 int2   to_i2 (uchar2 v) { return ( int2 )(v.x, v.y); }
 int2   to_i2 (uint2  v) { assert(all(v < INT_MAX));
                           return ( int2 )(v.x, v.y); }
uint2   to_ui2(uchar2 v) { return (uint2 )(v.x, v.y); }
uchar2  to_uc2(uint2  v) { assert(all(v < 256));
                           return (uchar2)(v.x, v.y); }
uchar2  to_uc2( int2  v) { assert(all(0 <= v)); assert(all(v < 256));
                           return (uchar2)(v.x, v.y); }

size_t  dim_size(size_t sz) { return sz; }
size_t  dim_size(uchar2 sz) { return sz.x * sz.y; }
size_t  dim_size(uint2  sz) { return sz.x * sz.y; }
size_t  dim_size(uint3  sz) { return sz.x * sz.y * sz.z; }


bool    dim_in_bounds(uint2 sz, uint2 i) { return all(i < sz); }
bool    dim_in_bounds( int2 sz,  int2 i) {
    assert(all(0 <= sz));
    return all((0 <= i) && (i < sz));
}

size_t  dim_idx (size_t sz, size_t i) { return i                                   ; }
size_t  dim_idx (uint2  sz, uint2  i) { assert(all(i < sz));
                                        return i.x + i.y * sz.x                    ; }
size_t  dim_idx (uchar2 sz, uchar2 i) { assert(all(i < sz));
                                        return dim_idx(to_ui2(sz), to_ui2(i))      ; }
size_t  dim_idx (uint3  sz, uint3  i) { assert(all(i < sz)); 
                                        return i.x + i.y * sz.x + i.z * sz.x * sz.y; }

uint2   dim_pos(uint2  sz, size_t idx) { assert(idx < dim_size(sz));
                                         return (uint2 )(idx % sz.x, idx / sz.x); }
uchar2  dim_pos(uchar2 sz, ushort idx) { assert(idx < dim_size(sz)); 
                                         return (uchar2)(idx % sz.x, idx / sz.x); }

// unpadded size & coord -> padded idx
size_t dim_idx_padded(uint padding, uint2 size, uint2 pos)
{ return dim_idx(padding * 2 + size, padding + pos); }

size_t dim_idx_padded(uint padding, uint2 size,  int2 pos) {
    int const padding_i = (int)padding;
    assert(all(-padding_i <=       pos              ));
    //if (!all(pos       <  to_i2(size + padding_i)))
    //    printf("%d, %d  of  %d, %d", pos.x, pos.y, to_i2(size + padding_i).x, to_i2(size + padding_i).y);
    assert(all( pos       <  to_i2(size + padding_i)));
    return dim_idx(padding * 2 + size, as_uint2(padding_i + pos));
}

