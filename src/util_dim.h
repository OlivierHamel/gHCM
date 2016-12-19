#pragma once


template<typename T, glm::precision P>
bool    dim_in_bounds(glm::tvec2<T, P> sz, glm::tvec2<T, P> i) {
    assert(glm::all(glm::lessThanEqual(glm::tvec2<T, P>(0), sz)));

    return glm::all(glm::lessThanEqual(glm::tvec2<T, P>(0),  i))
        && glm::all(glm::lessThan     (                 i , sz));
}

template<typename T, glm::precision P>
size_t  dim_size(glm::tvec2<T, P> const sz) {
    assert(glm::all(glm::lessThanEqual(glm::tvec2<T, P>(0), sz)));
    return sz.x * sz.y;
}

template<typename T, glm::precision P>
size_t  dim_idx(glm::tvec2<T, P> const sz, glm::tvec2<T, P> const i) {
    assert(dim_in_bounds(sz, i));
    return i.x + i.y * sz.x;
}

template<typename T, glm::precision P>
glm::tvec2<T, P> dim_pos(glm::tvec2<T, P> const sz, size_t const i) {
    assert(i < dim_size(sz));
    return { i % sz.x, i / sz.x };
}

