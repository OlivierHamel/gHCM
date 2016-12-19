
#include "eikonal.h"
#include "util_field.h"


using namespace glm;
using  int2  = glm::i32vec2;
using uint2  = glm::u32vec2;
using uchar2 = glm::u8vec2;


EikonalResult eikonal_fmm_sequential(Field2D<float>     const& field_cost
                                    ,std::vector<uint2> const& seed_points) {
    struct CellRef {
        int2 pos; float last_seen;

        bool operator<(CellRef const& b) const
        { return last_seen > b.last_seen; }
    };

    Field2D<float> field_time(field_cost.size(), INFINITY);

    auto const time   = [&](int2 const p) { return field_time.get_or(INFINITY, p); };
    auto const cost   = [&](int2 const p) { return field_cost.get_or(INFINITY, p); };
    auto const solve  = [&](int2 const p) {
        auto const cost_ = cost(p);
        assert(cost_   <  INFINITY);
        assert(time(p) == INFINITY); // shouldn't be asking to solve a cell once its locked

        auto const time_axis = [&](int2 const axis) { return min(time(p + axis), time(p - axis)); };
        float      dim[]     = { time_axis(int2(1, 0)), time_axis(int2(0, 1)) };
        std::sort(dim, dim + length_of(dim));
        return eikonal_solve(dim, cost_, INFINITY);
    };

    
    std::priority_queue<CellRef> open_queue;

    auto const chrono_bgn = std::chrono::high_resolution_clock::now();
    for (auto&& p : seed_points) {
        assert(field_time.in_bounds(p));
        open_queue.push({ int2(p), 0 });
    }

    while (!open_queue.empty()) {
        auto const curr = open_queue.top(); open_queue.pop();

        if (field_time[curr.pos] <= curr.last_seen) continue; // already processed.
        field_time[curr.pos] = curr.last_seen;

        auto const add_neighbour      = [&](int2 const offset) {
            auto const pos = curr.pos + offset;
            if ((cost(pos) < INFINITY) && (time(pos) == INFINITY))
                open_queue.push({ pos, solve(pos) });
        };
        auto const add_neighbour_axis = [&](int2 const axis  ) {
            add_neighbour(-axis);
            add_neighbour( axis);
        };

        add_neighbour_axis(int2(1, 0));
        add_neighbour_axis(int2(0, 1));
    }

    auto const chrono_end = std::chrono::high_resolution_clock::now();

    EikonalResult result;
    result.duration_total = result.duration_device = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_bgn);
    result.field = std::move(field_time);
    return result;
}

