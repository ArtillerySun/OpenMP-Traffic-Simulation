#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <math.h>
#include "common.h"

namespace traffic_prng {
    extern PRNG* engine;
}
    // on the same lane, find the position of the immediately next car
static inline int find_next(const std::vector<int>& lane, const int lane_num, const int L, int position) {
        // min 1 car next
    for (int i = 1; i < L; i++) {
            // mod to wrap back 
        int wrapped_shifted = (position + i) % L;
        if (lane[lane_num * L + wrapped_shifted] != -1) {
            return wrapped_shifted;
        } 
    }
    return -1; // not supposed to happen
}
static inline int find_prev(const std::vector<int>& lane,const int lane_num, const int L, int position) {
    for (int i = 1; i < L; i++) {
        int wrapped_shifted = (position - i + L) % L;
        if (lane[lane_num * L + wrapped_shifted] != -1) {
            return wrapped_shifted;
        } 
    }
    return -1; // not supposed to happen
}
    // compute distance from 2 position on circular road
static inline int dist(int L, int prev, int next) { 
    int d = (next - prev + L) % L; 
    return d == 0 ? L : d; 
}

static inline int safe_dist_next(const std::vector<int>& lane, const int lane_num, const int L, int position) {
    int p = find_next(lane, lane_num, L, position);
    return (p < 0) ? L : dist(L, position, p);
}

bool can_switch_lane(const std::vector<Car>& cars, const Car& c, const std::vector<int>& lanes, const int L) {
    int p2 = find_next(lanes, c.lane, L, c.position);
    int p3 = find_next(lanes, c.lane^1, L, c.position);
    int p0 = find_prev(lanes, c.lane^1, L, c.position);
    int d2 = (p2 < 0) ? L : dist(L, c.position, p2);
    int d0 = (p0 < 0) ? L : dist(L, p0, c.position);
    int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
    return d2 < d3 && 
            c.v >= d2 && 
            lanes[(c.lane^1) * L + c.position] == -1 
            && (d0 == L || d0 > cars[lanes[(c.lane^1) * L + p0]].v);
}

bool decelerate(const std::vector<Car>& cars,const std::vector<int>& lane, const int lane_num, const Car& c, const int L, 
        int* speed_snapshot, const int VMAX) {
    int p2 = find_next(lane, lane_num, L, c.position);
    int d  = (p2 < 0) ? L : dist(L, c.position, p2);
    int v2 = (p2 < 0) ? VMAX : cars[lane[lane_num * L + p2]].v; 
    int v1 = c.v;
    if (d <= v1) {
        if (v1 < v2 || v1 < 2) {
            *speed_snapshot = d - 1;
            return true;
        } else if (v1 >= v2 && v1 >= 2) {
            *speed_snapshot = std::min(d - 1, v1 - 2);
            return true;
        } else {
            return false;
        }
    }
    if (v1 < d && d <= 2 * v1 && v1 >= v2) {
        *speed_snapshot = v1 - std::floor((v1 - v2) / 2);
        return true;
    }
    return false;
}

void accelerate(const std::vector<int>& lane, const int lane_num, const Car& c, const int L, 
        int* speed_snapshot, const int& VMAX) {
    int p2 = find_next(lane, lane_num, L, c.position);
    int d = (p2 < 0) ? L : dist(L, c.position, p2);
    int v1 = c.v;
    *speed_snapshot = std::min(d - 1, std::min(v1 + 1, VMAX));
}

void executeSimulation(Params params, std::vector<Car> cars) {
    const int N = params.n;
    const int L = params.L;
    const int T = params.steps;
    const int VMAX = params.vmax;
    const double P_START = params.p_start;
    const double P_DEC = params.p_dec;
    int t = 0;
    std::vector<char> force_acc(N, 0);
    std::vector<int> cur_lanes(2 * L, -1); 
    std::vector<int> nxt_lanes(2 * L, -1); 

    // std::vector<int> speed_snapshot(N, 0);
    // std::vector<char> dec1(N, 0);

    std::vector<Car> tmp_cars(N);

    for (auto c : cars) {
        cur_lanes[c.lane * L + c.position] = c.id;
        // speed_snapshot[c.id] = c.v;
    }

    #ifdef DEBUG
    reportResult(cars, 0);
    #endif

    int num_threads = std::min(N, omp_get_max_threads());
    unsigned K = 0;

    while (t < T) {
        PRNG base = *traffic_prng::engine;
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int chunk_size = (N + num_threads - 1) / num_threads;
            int start = tid * chunk_size;
            int end = std::min(N, start + chunk_size);

            for (int i = start; i < end; i++) {
                const Car &c = cars[i];
                Car new_c = c;

                if (can_switch_lane(cars, c, cur_lanes, L)) {
                    new_c.lane ^= 1;
                }

                nxt_lanes[new_c.lane * L + new_c.position] = new_c.id;
                tmp_cars[i] = new_c;
            }
        }

        cur_lanes.swap(nxt_lanes);
        cars.swap(tmp_cars);
        std::fill(nxt_lanes.begin(), nxt_lanes.end(), -1);

        // std::cerr << "OK\n";

        #pragma omp parallel num_threads(num_threads) 
        {
            
            int tid = omp_get_thread_num();
            int chunk_size = (N + num_threads - 1) / num_threads;
            int start = tid * chunk_size;
            int end = std::min(N, start + chunk_size);

            PRNG e = base;
            e.discard(K + 2 * start);

            for (int i = start; i < end; i++) {
                Car &c = cars[i];
                Car new_c = c;
                

                bool if_acc = false;
                int* cur_speed = &(new_c.v);
                
                bool ss = flip_coin(P_START, &e);
                bool dec = flip_coin(P_DEC, &e);

                int d2 = safe_dist_next(cur_lanes, c.lane, L, c.position);
                if (c.v == 0 && d2 > 1) {   // satisfy slow start criteria
                    if (force_acc[c.id]) {  // did not accelerate when permitted last round
                        force_acc[c.id] = 0;
                        new_c.v = 1;
                    } 
                    else if (ss) {         // random start
                        accelerate(cur_lanes, c.lane, c, L, cur_speed, VMAX);
                        if_acc = 1;
                    } else {
                        new_c.v = 0;
                        force_acc[c.id] = 1;
                    }
                } else if (!if_acc && !force_acc[c.id] 
                    && !decelerate(cars, cur_lanes, c.lane, c, L, cur_speed, VMAX)) {
                    accelerate(cur_lanes, c.lane, c, L, cur_speed, VMAX);
                }

                if (dec) {
                    new_c.v = std::max(0, new_c.v - 1);
                }

                new_c.position = (new_c.position + new_c.v) % L;
                tmp_cars[i] = new_c;
                nxt_lanes[new_c.lane * L + new_c.position] = new_c.id;
            }
        }


        cur_lanes.swap(nxt_lanes);
        cars.swap(tmp_cars);
        std::fill(nxt_lanes.begin(), nxt_lanes.end(), -1);
        
        t++;
        K += 2*N;
    }

    #ifdef DEBUG
        reportFinalResult(cars);
    #endif
}