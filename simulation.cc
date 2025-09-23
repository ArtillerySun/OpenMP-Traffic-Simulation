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

static inline int find_next(const std::vector<int>& lane_pos, int position) {
    if (lane_pos.empty()) {
        return -1;
    }
    auto it = std::upper_bound(lane_pos.begin(), lane_pos.end(), position);
    
    if (it != lane_pos.end()) {
        return *it;
    }
    return lane_pos.front(); // Wrap around
}

static inline int find_prev(const std::vector<int>& lane_pos, int position) {
    if (lane_pos.empty()) {
        return -1;
    }
    auto it = std::lower_bound(lane_pos.begin(), lane_pos.end(), position);

    if (it != lane_pos.begin()) {
        return *(--it);
    }
    return lane_pos.back(); // Wrap around
}

static inline int dist(int L, int prev, int next) { 
    int d = (next - prev + L) % L; 
    return d == 0 ? L : d; 
}

static inline bool is_pos_occupied(const std::vector<int>& lane_pos, int pos) {
    auto it = std::lower_bound(lane_pos.begin(), lane_pos.end(), pos);
    return (it != lane_pos.end() && (*it) == pos);
}


void executeSimulation(Params params, std::vector<Car> cars) {
    const int N = params.n;
    const int L = params.L;
    const int T = params.steps;
    const int VMAX = params.vmax;
    const double P_START = params.p_start;
    const double P_DEC = params.p_dec;


    std::vector<std::vector<int>> lanes_pos(2);
    std::vector<std::vector<int>> lanes_id(2, std::vector<int>(L, -1));

    std::vector<bool> ss(N);
    std::vector<bool> dec(N);


    lanes_pos[0].reserve(N);
    lanes_pos[1].reserve(N);

    for (auto &c : cars) {
        lanes_pos[c.lane].push_back(c.position);
        lanes_id[c.lane][c.position] = c.id;
    }

    std::sort(lanes_pos[0].begin(), lanes_pos[0].end());
    std::sort(lanes_pos[1].begin(), lanes_pos[1].end());


    std::vector<int> next_speeds(N);
    std::vector<bool> change_lane_decisions(N, false);
    std::vector<bool> force_acc(N, 0);

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_lanes_pos_0(max_threads);
    std::vector<std::vector<int>> local_lanes_pos_1(max_threads);
    
    // std::vector<PRNG> thread_prngs;
    // int max_threads = omp_get_max_threads();

    // for (int i = 0; i < max_threads; ++i) {
    //     thread_prngs.emplace_back(params.seed + i);
    // }

    // PRNG base = *traffic_prng::engine;

    // unsigned K = 0;
    #pragma omp parallel
    for (int t = 0; t < T; t++) {

        #pragma omp single
        for (int i = 0; i < N; i++) {
            ss[i] = flip_coin(P_START, traffic_prng::engine);
            dec[i] = flip_coin(P_DEC, traffic_prng::engine);
        }

        //lane change
        #pragma omp for
        for (int i = 0; i < N; i++) {
            const Car& c = cars[i];
            const auto& current_lane_pos = lanes_pos[c.lane];
            const auto& other_lane_pos = lanes_pos[c.lane ^ 1];
            // const auto& current_lane_id = lanes_id[c.lane];
            const auto& other_lane_id = lanes_id[c.lane ^ 1];

            change_lane_decisions[c.id] = false;

            if (is_pos_occupied(other_lane_pos, c.position)) {
                continue;
            }

            int p2 = find_next(current_lane_pos, c.position);

            int d2 = (p2 < 0) ? L : dist(L, c.position, p2);

            if (c.v >= d2) {
                int p3 = find_next(other_lane_pos, c.position);
                int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
                if (d2 < d3) {
                    int p0 = find_prev(other_lane_pos, c.position);
                    int d0 = (p0 < 0) ? L : dist(L, p0, c.position);
                    if (p0 < 0 || d0 > cars[other_lane_id[p0]].v) {
                        change_lane_decisions[c.id] = true;
                    }
                }
            }
        }

        // apply lane change
        #pragma omp single
        {
            std::vector<int> moved_from_0_to_1;
            std::vector<int> moved_from_1_to_0;

            for (int i = 0; i < N; i++) {
                if (change_lane_decisions[i]) {
                    Car& c = cars[i];
                    if (c.lane == 0) {
                        moved_from_0_to_1.push_back(c.position);
                    } else {
                        moved_from_1_to_0.push_back(c.position);
                    }
                    lanes_id[c.lane][c.position] = -1;
                    c.lane ^= 1;
                    lanes_id[c.lane][c.position] = c.id;
                }
            }

            if (!moved_from_0_to_1.empty() || !moved_from_1_to_0.empty()) {
                
                if (!moved_from_0_to_1.empty()) {
                    std::sort(moved_from_0_to_1.begin(), moved_from_0_to_1.end());
                    auto new_end = std::remove_if(lanes_pos[0].begin(), lanes_pos[0].end(), [&](int pos) {
                        return std::binary_search(moved_from_0_to_1.begin(), moved_from_0_to_1.end(), pos); 
                    });
                    lanes_pos[0].erase(new_end, lanes_pos[0].end());
                }

                if (!moved_from_1_to_0.empty()) {
                    std::sort(moved_from_1_to_0.begin(), moved_from_1_to_0.end());
                    auto new_end = std::remove_if(lanes_pos[1].begin(), lanes_pos[1].end(), [&](int pos) {
                        return std::binary_search(moved_from_1_to_0.begin(), moved_from_1_to_0.end(), pos);
                    });
                    lanes_pos[1].erase(new_end, lanes_pos[1].end());
                }

                std::vector<int> new_lane0;
                new_lane0.reserve(lanes_pos[0].size() + moved_from_1_to_0.size());
                std::merge(lanes_pos[0].begin(), lanes_pos[0].end(),
                        moved_from_1_to_0.begin(), moved_from_1_to_0.end(),
                        std::back_inserter(new_lane0));
                lanes_pos[0] = std::move(new_lane0);

                std::vector<int> new_lane1;
                new_lane1.reserve(lanes_pos[1].size() + moved_from_0_to_1.size());
                std::merge(lanes_pos[1].begin(), lanes_pos[1].end(),
                        moved_from_0_to_1.begin(), moved_from_0_to_1.end(),
                        std::back_inserter(new_lane1));
                lanes_pos[1] = std::move(new_lane1);
            }
        }

        // acc & dec
        #pragma omp for
        for (int i = 0; i < N; i++) {
            const Car& c = cars[i];
            const auto& current_lane_pos = lanes_pos[c.lane];
            const auto& current_lane_id = lanes_id[c.lane];
            
            int p2 = find_next(current_lane_pos, c.position);
            int d = (p2 < 0) ? L : dist(L, c.position, p2);
            
            
            // unsigned int k = K + 2 * i;
            // PRNG e = base;
            // e.discard(k);
            // bool ss = flip_coin(P_START, &e);
            // bool dec = flip_coin(P_DEC, &e);
        

            int new_v = c.v;
            bool skip_2 = false, skip_3 = false;
            // slow start
            if (c.v == 0 && d > 1) {
                if (force_acc[c.id]) {
                    new_v = 1;
                    force_acc[c.id] = 0;
                } else if (!ss[i]) {
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
                int d  = (p2 < 0) ? L : dist(L, c.position, p2);
                int v2 = (p2 < 0) ? VMAX : cars[current_lane_id[p2]].v; 
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
            if (dec[i]) {
                new_v = std::max(0, new_v - 1);
            }
            
            next_speeds[c.id] = new_v;
        }


        // update
        
        #pragma omp for
        for (int i = 0; i < N; i++) {
            auto& c = cars[i];
            int old_position = c.position;
            c.v = next_speeds[i];
            c.position = (old_position + c.v) % L;
        }

        int tid = omp_get_thread_num();
        local_lanes_pos_0[tid].clear();
        local_lanes_pos_1[tid].clear();

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            if (cars[i].lane == 0) {
                local_lanes_pos_0[tid].push_back(cars[i].position);
            } else {
                local_lanes_pos_1[tid].push_back(cars[i].position);
            }
        }

        std::sort(local_lanes_pos_0[tid].begin(), local_lanes_pos_0[tid].end());
        std::sort(local_lanes_pos_1[tid].begin(), local_lanes_pos_1[tid].end());

        #pragma omp barrier

        #pragma omp single
        {
            lanes_id[0].assign(L, -1);
            lanes_id[1].assign(L, -1);
            for(auto& c : cars) {
                lanes_id[c.lane][c.position] = c.id;
            }

            // lane 0
            lanes_pos[0].clear();
            for(int i = 0; i < max_threads; ++i) {
                lanes_pos[0].insert(lanes_pos[0].end(), local_lanes_pos_0[i].begin(), local_lanes_pos_0[i].end());
            }
            std::sort(lanes_pos[0].begin(), lanes_pos[0].end());

            // lane 1
            lanes_pos[1].clear();
            for(int i = 0; i < max_threads; ++i) {
                lanes_pos[1].insert(lanes_pos[1].end(), local_lanes_pos_1[i].begin(), local_lanes_pos_1[i].end());
            }
            std::sort(lanes_pos[1].begin(), lanes_pos[1].end());
        }
    }

    #ifdef DEBUG
    reportFinalResult(cars);
    #endif
}