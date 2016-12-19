
#include "eikonal.h"
#include "util_field.h"
#include "bitset_dynamic.h"

namespace {

int2 const k_border_directions[] { int2(-1, 0), int2(0, -1), int2(1, 0), int2(0, 1) };

}


EikonalResult eikonal_fim_sequential(float              const  minimum_change
                                    ,Field2D<float>     const& field_cost
                                    ,std::vector<uint2> const& seed_points) {
    Field2D<float>    field_time(field_cost.size(), INFINITY);
    // INVARIANT: open_set/open_list are mirrors.
    BitsetDynamic     open_set  (field_cost.size_flat());
    std::deque<int2>  open_list;


    auto const time   = [&](int2 const p) { return field_time.get_or(INFINITY, p); };
    auto const cost   = [&](int2 const p) { return field_cost.get_or(INFINITY, p); };
    auto const open   = [&](int2 const p) { return open_set[field_time.size().x * p.y + p.x]; };
    auto const solve  = [&](int2 const p) {
        assert(field_time.in_bounds(p));
        if (!field_time.in_bounds(p)) return;

        struct SolverResult { float time_old, time_new, time_diff; };

        auto  const eikonal_solve_helper = [&](int2 const pos) {
            auto  const cost_     = cost(pos); assert(cost_ <  INFINITY);
            auto  const time_axis = [&](int2 const axis) { return min(time(pos + axis)
                                                                     ,time(pos - axis)); };
            float       dim[]     = { time_axis(int2(1, 0)), time_axis(int2(0, 1)) };
            std::sort(dim, dim + length_of(dim));

            SolverResult r;
            r.time_old  = time(pos);
            r.time_new  = eikonal_solve(dim, cost_, r.time_old);
            r.time_diff = eikonal_time_diff(r.time_new, r.time_old);
            return r;
        };

        auto const result = eikonal_solve_helper(p);
        field_time[p] = result.time_new;
        
        // if we haven't converged -> re-enqueue
        if (minimum_change < std::abs(result.time_diff)) {
            open_list.push_back(p);
            return;
        }

        // if we've converged, check our neighbours
        auto const add_neighbour      = [&](int2 const offset) {
            auto const pos  = p + offset;
            if (cost(pos) == INFINITY ) return;
            if (open(pos)             ) return;

            auto const result = eikonal_solve_helper(pos);
            if (result.time_diff < 0) { // if we've improved then reconsider...
                field_time[pos] = result.time_new;

                assert(std::find(open_list.begin(), open_list.end(), pos) == open_list.end());
                open(pos) = true;
                open_list.push_back(pos);
            }
        };
        auto const add_neighbour_axis = [&](int2 const axis  ) {
            add_neighbour(-axis);
            add_neighbour( axis);
        };

        add_neighbour_axis(int2(1, 0));
        add_neighbour_axis(int2(0, 1));

        // remove ourselves from the open set since we've converged
        open(p) = false;
    };


    auto const chrono_bgn = std::chrono::high_resolution_clock::now();
    for (auto&& p : seed_points) {
        field_time[p] = 0;

        for (auto&& dir : k_border_directions) {
            auto const pos          = int2(p) + dir;
            if (!all(lessThanEqual(int2(0), pos                    ))) continue;
            if (!all(lessThan     (pos    , int2(field_time.size())))) continue;

            open(pos) = true;
            open_list.push_back(pos);
        }
    }

    while (!open_list.empty()) {
        auto const curr = open_list.front(); open_list.pop_front();
        solve(curr);
    }

    auto const chrono_end = std::chrono::high_resolution_clock::now();

    EikonalResult result;
    result.duration_total = result.duration_device = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_bgn);
    result.field = std::move(field_time);
    return result;
}

