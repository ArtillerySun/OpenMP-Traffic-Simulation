#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <set>
#include <unordered_map>
#include <math.h>
#include "common.h"

namespace traffic_prng {
    extern PRNG* engine;
}

struct CarPositionComparator {
    bool operator()(const Car* a, int pos) const {
        return a->position < pos;
    }
    bool operator()(int pos, const Car* a) const {
        return pos < a->position;
    }
};

static inline const Car* find_next(const std::vector<Car*>& lane, int position) {
    if (lane.empty()) {
        return nullptr;
    }
    auto it = std::upper_bound(lane.begin(), lane.end(), position, CarPositionComparator());
    
    if (it != lane.end()) {
        return *it;
    }
    return lane.front(); // Wrap around
}

static inline const Car* find_prev(const std::vector<Car*>& lane, int position) {
    if (lane.empty()) {
        return nullptr;
    }
    auto it = std::lower_bound(lane.begin(), lane.end(), position, CarPositionComparator());

    if (it != lane.begin()) {
        return *(--it);
    }
    return lane.back(); // Wrap around
}

static inline int dist(int L, int prev, int next) { 
    int d = (next - prev + L) % L; 
    return d == 0 ? L : d; 
}

static inline bool is_pos_occupied(const std::vector<Car*>& lane, int pos) {
    auto it = std::lower_bound(lane.begin(), lane.end(), pos, CarPositionComparator());
    return (it != lane.end() && (*it)->position == pos);
}


void executeSimulation(Params params, std::vector<Car> cars) {
    const int N = params.n;
    const int L = params.L;
    const int T = params.steps;
    const int VMAX = params.vmax;
    const double P_START = params.p_start;
    const double P_DEC = params.p_dec;


    std::vector<std::vector<Car*>> lanes(2);

    lanes[0].reserve(N);
    lanes[1].reserve(N);


    for (int i = 0; i < N; i++) {
        lanes[cars[i].lane].push_back(&cars[i]);
    }
    std::sort(lanes[0].begin(), lanes[0].end(), [](const Car* a, const Car* b){ return a->position < b->position; });
    std::sort(lanes[1].begin(), lanes[1].end(), [](const Car* a, const Car* b){ return a->position < b->position; });


    std::vector<int> next_speeds(N);
    std::vector<bool> change_lane_decisions(N, false);
    std::vector<bool> force_acc(N, 0);
    
    PRNG base = *traffic_prng::engine;

    #pragma omp parallel
    for (int t = 0, K = 0; t < T; t++, K += 2 * N) {

        //lane change
        #pragma omp for
        for (int i = 0; i < N; i++) {
            const Car& c = cars[i];
            const auto& current_lane = lanes[c.lane];
            const auto& other_lane = lanes[c.lane ^ 1];

            change_lane_decisions[c.id] = false;

            if (is_pos_occupied(other_lane, c.position)) {
                continue;
            }

            const Car* p2 = find_next(current_lane, c.position);

            int d2 = (p2 == nullptr) ? L : dist(L, c.position, p2->position);

            if (c.v >= d2) {
                const Car* p3 = find_next(other_lane, c.position);
                int d3 = (p3 == nullptr) ? L : dist(L, c.position, p3->position);
                if (d2 < d3) {
                    const Car* p0 = find_prev(other_lane, c.position);
                    int d0 = (p0 == nullptr) ? L : dist(L, p0->position, c.position);
                    if (p0 == nullptr || d0 > p0->v) {
                        change_lane_decisions[c.id] = true;
                    }
                }
            }
        }

        // apply lane change
        bool has_lane_changed = false;

        #pragma omp single
        {
            for (int i = 0; i < N; ++i) {
                if (change_lane_decisions[i]) {
                    has_lane_changed = true;
                    Car& c = cars[i];
                    c.lane ^= 1;
                }
            }
            
            if (has_lane_changed) {
                lanes[0].clear();
                lanes[1].clear();
                for (int i = 0; i < N; ++i) {
                    lanes[cars[i].lane].push_back(&cars[i]);
                }
                std::sort(lanes[0].begin(), lanes[0].end(), [](const Car* a, const Car* b){ return a->position < b->position; });
                std::sort(lanes[1].begin(), lanes[1].end(), [](const Car* a, const Car* b){ return a->position < b->position; });
            }
        }

        // acc & dec
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            const Car& c = cars[i];
            const auto& current_lane = lanes[c.lane];
            
            const Car* next_car = find_next(current_lane, c.position);
            int d = (next_car == nullptr) ? L : dist(L, c.position, next_car->position);
            
            
            long long k = K + 2 * i;
            PRNG e = base;
            e.discard(k);
            bool ss = flip_coin(P_START, &e);
            bool dec = flip_coin(P_DEC, &e);
        

            int new_v = c.v;
            bool skip_2 = false, skip_3 = false;
            // slow start
            if (c.v == 0 && d > 1) {
                if (force_acc[c.id]) {
                    new_v = 1;
                    force_acc[c.id] = 0;
                } else if (!ss) {
                    new_v = 0;
                    force_acc[c.id] = 1;
                    skip_2 = true;
                    skip_3 = true;
                } else { // ss = 1
                    skip_2 = true;
                }
            }

            // dec
            bool modified = false;
            if (!skip_2) {
                int p2 = next_car->position;
                int d  = (p2 < 0) ? L : dist(L, c.position, p2);
                int v2 = (p2 < 0) ? VMAX : next_car->v; 
                int v1 = c.v;
                if (d <= v1) {
                    if (v1 < v2 || v1 < 2) {
                        new_v = d - 1;
                        modified = true;
                    } else if (v1 >= v2 && v1 >= 2) {
                        new_v = std::min(d - 1, v1 - 2);
                        modified = true;
                    }
                }
                if (v1 < d && d <= 2 * v1 && v1 >= v2) {
                    new_v = v1 - (v1 - v2) / 2;
                    modified = true;
                }
            }

            // acc
            if (!skip_3 && !modified) {
                new_v = std::min(d - 1, std::min(c.v + 1, VMAX));
            }
            
            // dec with probabilty
            if (dec) {
                new_v = std::max(0, new_v - 1);
            }
            
            next_speeds[c.id] = new_v;
        }


        // update
        #pragma omp single
        {
            for (int i = 0; i < N; ++i) {
                cars[i].v = next_speeds[i];
                cars[i].position = (cars[i].position + cars[i].v) % L;
            }

            std::sort(lanes[0].begin(), lanes[0].end(), [](const Car* a, const Car* b){ return a->position < b->position; });
            std::sort(lanes[1].begin(), lanes[1].end(), [](const Car* a, const Car* b){ return a->position < b->position; });
        }
    }

    #ifdef DEBUG
    reportFinalResult(cars);
    #endif
}