#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "common.h"

namespace traffic_prng {
    extern PRNG* engine;
}

static auto sortByPosition = [](const Car* a, const Car* b) {
    return a->position < b->position;
};

static inline Car* find_next_ptr(const std::vector<Car*>& lane_ptrs, int position) {
    if (lane_ptrs.empty()) return nullptr;
    auto it = std::upper_bound(lane_ptrs.begin(), lane_ptrs.end(), position, 
        [](int pos, const Car* c){ return pos < c->position; });
    return (it != lane_ptrs.end()) ? *it : lane_ptrs.front();
}

static inline Car* find_prev_ptr(const std::vector<Car*>& lane_ptrs, int position) {
    if (lane_ptrs.empty()) return nullptr;
    auto it = std::lower_bound(lane_ptrs.begin(), lane_ptrs.end(), position,
        [](const Car* c, int pos){ return c->position < pos; });
    return (it != lane_ptrs.begin()) ? *(--it) : lane_ptrs.back();
}

static inline bool is_pos_occupied_ptr(const std::vector<Car*>& lane_ptrs, int pos) {
    auto it = std::lower_bound(lane_ptrs.begin(), lane_ptrs.end(), pos,
        [](const Car* c, int val){ return c->position < val; });
    return (it != lane_ptrs.end() && (*it)->position == pos);
}

static inline int dist(int L, int prev, int next) { 
    int d = (next - prev + L) % L; 
    return d == 0 ? L : d; 
}


static void parallel_merge(std::vector<Car*>& final_array, 
    std::vector<std::vector<Car*>>& to_merge, int max_threads) {
    
    if (max_threads == 0) { final_array.clear(); return; }
    if (max_threads == 1) {
        final_array.swap(to_merge[0]);
        return;
    }

    int num_active_lanes = max_threads;
    int group_size = 2;

    while (num_active_lanes > 1) {
        #pragma omp taskgroup
        {
            for (int i = 0; i + (group_size >> 1) < num_active_lanes; i += group_size) {
                #pragma omp task
                {
                    std::vector<Car*> temp_buffer;
                    temp_buffer.reserve(to_merge[i].size() + to_merge[i + (group_size >> 1)].size());
                    std::merge(to_merge[i].begin(), to_merge[i].end(),
                               to_merge[i + (group_size >> 1)].begin(), to_merge[i + (group_size >> 1)].end(),
                               std::back_inserter(temp_buffer), sortByPosition);
                    to_merge[i] = std::move(temp_buffer);
                }
            }
        }
        #pragma omp taskwait

        if (num_active_lanes % 2 != 0) {
            to_merge[num_active_lanes >> 1] = std::move(to_merge[num_active_lanes - 1]);
        }

        num_active_lanes = (num_active_lanes + 1) >> 1;
        group_size <<= 1;
    }

    final_array.swap(to_merge[0]);
}


void executeSimulation(Params params, std::vector<Car> cars) {
    const int N = params.n, L = params.L, T = params.steps, VMAX = params.vmax;
    const double P_START = params.p_start, P_DEC = params.p_dec;

    std::vector<std::vector<Car*>> lanes_car_ptrs(2);
    lanes_car_ptrs[0].reserve(N);
    lanes_car_ptrs[1].reserve(N);

    for (int i = 0; i < N; ++i) {
        lanes_car_ptrs[cars[i].lane].push_back(&cars[i]);
    }
    std::sort(lanes_car_ptrs[0].begin(), lanes_car_ptrs[0].end(), sortByPosition);
    std::sort(lanes_car_ptrs[1].begin(), lanes_car_ptrs[1].end(), sortByPosition);

    std::vector<int> next_speeds(N);
    std::vector<bool> change_lane_decisions(N, false);
    std::vector<bool> force_acc(N, 0);
    std::vector<bool> ss(N), dec(N);
    
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<Car*>> local_lanes_ptrs_0(max_threads);
    std::vector<std::vector<Car*>> local_lanes_ptrs_1(max_threads);

    #pragma omp parallel
    for (int t = 0; t < T; t++) {

        // prng
        #pragma omp single
        for(int i = 0; i < N; i++) { 
            ss[i] = flip_coin(P_START, traffic_prng::engine); 
            dec[i] = flip_coin(P_DEC, traffic_prng::engine);
        }

        // change lane
        #pragma omp for
        for (int i = 0; i < N; i++) {
            Car& c = cars[i];
            const auto& current_lane_ptrs = lanes_car_ptrs[c.lane];
            const auto& other_lane_ptrs = lanes_car_ptrs[c.lane ^ 1];
            
            change_lane_decisions[c.id] = false;
            if (is_pos_occupied_ptr(other_lane_ptrs, c.position)) continue;

            Car* p2 = find_next_ptr(current_lane_ptrs, c.position);
            int d2 = (p2 == nullptr) ? L : dist(L, c.position, p2->position);

            if (c.v >= d2) {
                Car* p3 = find_next_ptr(other_lane_ptrs, c.position);
                int d3 = (p3 == nullptr) ? L : dist(L, c.position, p3->position);
                if (d2 < d3) {
                    Car* p0 = find_prev_ptr(other_lane_ptrs, c.position);
                    int d0 = (p0 == nullptr) ? L : dist(L, p0->position, c.position);
                    if (p0 == nullptr || d0 > p0->v) {
                        change_lane_decisions[c.id] = true;
                    }
                }
            }
        }
        
        // apply lane change
        #pragma omp single
        {
            std::vector<Car*> moved_from_0_to_1;
            std::vector<Car*> moved_from_1_to_0;

            for (int i = 0; i < N; i++) {
                if (change_lane_decisions[i]) {
                    Car& c = cars[i];
                    if (c.lane == 0) {
                        moved_from_0_to_1.push_back(&c);
                    } else {
                        moved_from_1_to_0.push_back(&c);
                    }
                    c.lane ^= 1;
                }
            }

            if (!moved_from_0_to_1.empty() || !moved_from_1_to_0.empty()) {           
                if (!moved_from_0_to_1.empty()) {
                    std::sort(moved_from_0_to_1.begin(), moved_from_0_to_1.end(), sortByPosition);
                    auto new_end = std::remove_if(lanes_car_ptrs[0].begin(), lanes_car_ptrs[0].end(), [&](Car* p) {
                        return std::binary_search(moved_from_0_to_1.begin(), moved_from_0_to_1.end(), p, sortByPosition); 
                    });
                    lanes_car_ptrs[0].erase(new_end, lanes_car_ptrs[0].end());
                }

                if (!moved_from_1_to_0.empty()) {
                    std::sort(moved_from_1_to_0.begin(), moved_from_1_to_0.end(), sortByPosition);
                    auto new_end = std::remove_if(lanes_car_ptrs[1].begin(), lanes_car_ptrs[1].end(), [&](Car* p) {
                        return std::binary_search(moved_from_1_to_0.begin(), moved_from_1_to_0.end(), p, sortByPosition);
                    });
                    lanes_car_ptrs[1].erase(new_end, lanes_car_ptrs[1].end());
                }

                std::vector<Car*> new_lane0;
                new_lane0.reserve(lanes_car_ptrs[0].size() + moved_from_1_to_0.size());
                std::merge(lanes_car_ptrs[0].begin(), lanes_car_ptrs[0].end(), moved_from_1_to_0.begin(), 
                    moved_from_1_to_0.end(), std::back_inserter(new_lane0), sortByPosition);
                lanes_car_ptrs[0] = std::move(new_lane0);

                std::vector<Car*> new_lane1;
                new_lane1.reserve(lanes_car_ptrs[1].size() + moved_from_0_to_1.size());
                std::merge(lanes_car_ptrs[1].begin(), lanes_car_ptrs[1].end(), moved_from_0_to_1.begin(), 
                    moved_from_0_to_1.end(), std::back_inserter(new_lane1), sortByPosition);
                lanes_car_ptrs[1] = std::move(new_lane1);
            }
        }

        // acc and dec
        #pragma omp for
        for (int i = 0; i < N; i++) {
            Car& c = cars[i];
            const auto& current_lane_ptrs = lanes_car_ptrs[c.lane];
            
            Car* p2 = find_next_ptr(current_lane_ptrs, c.position);
            int d = (p2 == nullptr) ? L : dist(L, c.position, p2->position);

            int new_v = c.v;
            bool modified = false;

            if (c.v == 0 && d > 1) {
                if (force_acc[c.id]) {
                    new_v = 1; force_acc[c.id] = false;
                } else if (!ss[c.id]) {
                    new_v = 0; force_acc[c.id] = true; modified = true;
                }
            }
            if (!modified) {
                int v1 = c.v;
                int v2 = (p2 == nullptr) ? VMAX : p2->v; 
                if (d <= v1) {
                    if (v1 < v2 || v1 < 2) new_v = d - 1;
                    else new_v = std::min(d - 1, v1 - 2);
                } else if (v1 < d && d <= 2 * v1 && v1 >= v2) {
                    new_v = v1 - (v1 - v2) / 2;
                } else {
                    new_v = std::min(d - 1, std::min(v1 + 1, VMAX));
                }
            }
            if (dec[c.id]) {
                new_v = std::max(0, new_v - 1);
            }
            next_speeds[c.id] = new_v;
        }

        // update
        #pragma omp for
        for (int i = 0; i < N; i++) {
            cars[i].v = next_speeds[i];
            cars[i].position = (cars[i].position + cars[i].v) % L;
        }

        #pragma omp barrier

        int tid = omp_get_thread_num();
        local_lanes_ptrs_0[tid].clear();
        local_lanes_ptrs_1[tid].clear();

        #pragma omp for
        for (int i = 0; i < N; i++) {
            int current_tid = omp_get_thread_num();
            if (cars[i].lane == 0) {
                local_lanes_ptrs_0[current_tid].push_back(&cars[i]);
            } else {
                local_lanes_ptrs_1[current_tid].push_back(&cars[i]);
            }
        }

        std::sort(local_lanes_ptrs_0[tid].begin(), local_lanes_ptrs_0[tid].end(), sortByPosition);
        std::sort(local_lanes_ptrs_1[tid].begin(), local_lanes_ptrs_1[tid].end(), sortByPosition);
        
        #pragma omp barrier

        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task
                parallel_merge(lanes_car_ptrs[0], local_lanes_ptrs_0, max_threads);
                #pragma omp task
                parallel_merge(lanes_car_ptrs[1], local_lanes_ptrs_1, max_threads);
            }
        }
    }
    #ifdef DEBUG
    reportFinalResult(cars);
    #endif
}