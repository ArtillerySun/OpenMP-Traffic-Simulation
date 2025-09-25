#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <math.h>
#include "common.h"
#include <cstring>

namespace traffic_prng {
    extern PRNG* engine;
}

static inline int find_next_ptr(const int *lane_ptr, const int L, int position) {
    for (int i = position + 1; i < L; i++) {
        if (lane_ptr[i] != -1) return i;
    }
    for (int i = 0; i < position; i++) {
        if (lane_ptr[i] != -1) return i;
    }
    return -1;
}
static inline int find_prev_ptr(const int *lane_ptr, const int L, int position) {
    for (int i = position - 1; i >= 0; i--) {
        if (lane_ptr[i] != -1) return i;
    }
    for (int i = L - 1; i > position; i--) {
        if (lane_ptr[i] != -1) return i;
    }
    return -1;
}
    // compute distance from 2 position on circular road
static inline int dist(int L, int prev, int next) { 
    int d = next - prev; 
    return d + (d < 0) * L; 
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

    std::vector<Car> tmp_cars(N);

    for (auto c : cars) {
        cur_lanes[c.lane * L + c.position] = c.id;
    }

    #ifdef DEBUG
    reportResult(cars, 0);
    #endif

    int num_threads = std::min(N, omp_get_max_threads());

    int chunk_size = (N + num_threads - 1) / num_threads;

    std::vector<int> st(num_threads), ed(num_threads);

    // #pragma omp parallel for
    for (int i = 0; i < num_threads; i ++) {
        st[i] = i * chunk_size;
        ed[i] = std::min(N, st[i] + chunk_size);
    }

    unsigned K = 0;

    while (t < T) {
        PRNG base = *traffic_prng::engine;
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int start = st[tid];
            int end = ed[tid];

            for (int i = start; i < end; i++) {
                const Car &c = cars[i];
                tmp_cars[i] = c;
                Car &new_c = tmp_cars[i];

                int other_lane = c.lane ^ 1;
                const int *lane_ptr = cur_lanes.data() + c.lane * L;
                const int *other_lane_ptr = cur_lanes.data() + other_lane * L;
                int p2 = find_next_ptr(lane_ptr, L, c.position);
                int p3 = find_next_ptr(other_lane_ptr, L, c.position);
                int p0 = find_prev_ptr(other_lane_ptr, L, c.position);


                int d2 = (p2 < 0) ? L : dist(L, c.position, p2);
                int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
                int d0 = (p0 < 0) ? L : dist(L, p0, c.position);
                
                if (d2 < d3 && 
                    c.v >= d2 && 
                    cur_lanes[other_lane * L + c.position] == -1 && 
                    (p0 < 0 || d0 > cars[cur_lanes[other_lane * L + p0]].v)) 
                {
                    new_c.lane ^= 1;
                }

                nxt_lanes[new_c.lane * L + new_c.position] = new_c.id;
            }
        }

        cur_lanes.swap(nxt_lanes);
        cars.swap(tmp_cars);
        std::memset(nxt_lanes.data(), 0xff, nxt_lanes.size() * sizeof(int));

        #pragma omp parallel num_threads(num_threads) 
        {
            
            int tid = omp_get_thread_num();
            int start = st[tid];
            int end = ed[tid];

            PRNG e = base;
            e.discard(K + 2 * start);

            for (int i = start; i < end; i++) {
                Car &c = cars[i];
                tmp_cars[i] = c;
                Car &new_c = tmp_cars[i];
                
                const int *lane_ptr = cur_lanes.data() + c.lane * L;
                int p2 = find_next_ptr(lane_ptr, L, c.position);
                int d = (p2 < 0) ? L : dist(L, c.position, p2);
                int v2 = (p2 < 0) ? VMAX : cars[cur_lanes[c.lane * L + p2]].v;

                bool if_acc = false;
                
                bool ss = flip_coin(P_START, &e);
                bool dec = flip_coin(P_DEC, &e);

                if (c.v == 0 && d > 1) {   // satisfy slow start criteria
                    if (force_acc[c.id]) {  // did not accelerate when permitted last round
                        force_acc[c.id] = 0;
                        new_c.v = 1;
                    } 
                    else if (ss) {         // random start
                        new_c.v = std::min(d - 1, std::min(c.v + 1, VMAX));
                        if_acc = 1;
                    } else {
                        new_c.v = 0;
                        force_acc[c.id] = 1;
                    }
                } else if (!if_acc && !force_acc[c.id]) {
                    if (d <= c.v) {
                        if (c.v < v2 || c.v < 2) {
                            new_c.v = d - 1;
                        } else if (c.v >= v2 && c.v >= 2) {
                            new_c.v = std::min(d - 1, c.v - 2);
                        }
                    } else if (d <= 2 * c.v && c.v >= v2) {
                        new_c.v = c.v - (c.v - v2) / 2;
                    } else {
                        new_c.v = std::min(d - 1, std::min(c.v + 1, VMAX));
                    }
                }

                if (dec) {
                    new_c.v = std::max(0, new_c.v - 1);
                }

                new_c.position = new_c.position + new_c.v;
                new_c.position = new_c.position >= L ? new_c.position - L : new_c.position;
                nxt_lanes[new_c.lane * L + new_c.position] = new_c.id;
            }
        }


        cur_lanes.swap(nxt_lanes);
        cars.swap(tmp_cars);
        std::memset(nxt_lanes.data(), 0xff, nxt_lanes.size() * sizeof(int));
        
        t++;
        K += 2*N;
    }

    #ifdef DEBUG
        reportFinalResult(cars);
    #endif
}