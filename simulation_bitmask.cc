#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <math.h>
#include <cstring>

#include "common.h"

namespace traffic_prng {
    extern PRNG* engine;
}

static inline int find_next_bitmask(const std::vector<uint64_t>& bitmask, int lane_num, int L, int position) {
    if (L <= 1) return -1;

    int offset = lane_num * L;
    
    if (position + 1 < L) { // pos + 1 to the end of the lane
        int start_bit = offset + position + 1;
        int start_word = start_bit / 64;
        start_bit %= 64;
        int end_word = (offset + L - 1) / 64;
        int end_bit = (offset + L - 1) % 64;

        if (start_word == end_word) {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit) & 
            (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return __builtin_ctzll(word) + start_word * 64 - offset;
            }
        } else {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit);
            if (word) {
                return __builtin_ctzll(word) + start_word * 64 - offset;
            }

            for (int idx = start_word + 1; idx < end_word; idx++) {
                word = bitmask[idx];
                if (word) {
                    return __builtin_ctzll(word) + idx * 64 - offset;
                }
            }

            word = bitmask[end_word] & (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return __builtin_ctzll(word) + end_word * 64 - offset;
            }
        }  
    }

    if (position) { // 0 to pos - 1
        int start_bit = offset;
        int start_word = start_bit / 64;
        start_bit %= 64;

        int end_bit = offset + position - 1;
        int end_word = end_bit / 64;
        end_bit %= 64;

        if (start_word == end_word) {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit) & 
            (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return __builtin_ctzll(word) + start_word * 64 - offset;
            }
        } else {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit);
            if (word) {
                return __builtin_ctzll(word) + start_word * 64 - offset;
            }

            for (int idx = start_word + 1; idx < end_word; idx++) {
                word = bitmask[idx];
                if (word) {
                    return __builtin_ctzll(word) + idx * 64 - offset;
                }
            }

            word = bitmask[end_word] & (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return __builtin_ctzll(word) + end_word * 64 - offset;
            }
        }
    }
    
    return -1;
}

static inline int find_prev_bitmask(const std::vector<uint64_t>& bitmask, int lane_num, int L, int position) {
    if (L <= 1) return -1;

    int offset = lane_num * L;

    if (position) { // 0 to pos - 1
        int start_bit = offset;
        int start_word = start_bit / 64;
        start_bit %= 64;

        int end_bit = offset + position - 1;
        int end_word = end_bit / 64;
        end_bit %= 64;

        if (start_word == end_word) {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit) & 
            (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return 63 - __builtin_clzll(word) + start_word * 64 - offset;
            }
        } else {

            uint64_t word = bitmask[end_word] & (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return 63 - __builtin_clzll(word) + end_word * 64 - offset;
            }

            for (int idx = end_word - 1; idx > start_word; idx--) {
                word = bitmask[idx];
                if (word) {
                    return 63 - __builtin_clzll(word) + idx * 64 - offset;
                }
            }

            word = bitmask[start_word] & (~0ULL << start_bit);
            if (word) {
                return 63 - __builtin_clzll(word) + start_word * 64 - offset;
            }
        }
    }

    if (position + 1 < L) { // pos + 1 to the end of the lane
        int start_bit = offset + position + 1;
        int start_word = start_bit / 64;
        start_bit %= 64;
        int end_word = (offset + L - 1) / 64;
        int end_bit = (offset + L - 1) % 64;

        if (start_word == end_word) {
            uint64_t word = bitmask[start_word] & (~0ULL << start_bit) & 
            (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return 63 - __builtin_clzll(word) + start_word * 64 - offset;
            }
        } else {
            uint64_t word = bitmask[end_word] & (end_bit == 63 ? ~0ULL : ~(~0ULL << (end_bit + 1)));
            if (word) {
                return 63 - __builtin_clzll(word) + end_word * 64 - offset;
            }
            
            for (int idx = end_word - 1; idx > start_word; idx--) {
                word = bitmask[idx];
                if (word) {
                    return 63 - __builtin_clzll(word) + idx * 64 - offset;
                }
            }

            word = bitmask[start_word] & (~0ULL << start_bit);
            if (word) {
                return 63 - __builtin_clzll(word) + start_word * 64 - offset;
            }
        }  
    }

    return -1;
}

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
    unsigned K = 0;

    std::vector<char> force_acc(N, 0);

    std::vector<int> cur_lanes(2 * L, -1);
    std::vector<uint64_t> lane_bitmask((2 * L + 63) / 64, 0);

    std::vector<int> nxt_lanes(2 * L, -1);
    std::vector<uint64_t> nxt_lane_bitmask((2 * L + 63) / 64, 0);

    std::vector<Car> tmp_cars(N);

    for (auto c : cars) {
        int lane = c.lane;
        int pos = c.position;
        int offset = lane * L + pos;
        cur_lanes[offset] = c.id;
        lane_bitmask[offset / 64] |= (1ull << (offset % 64));
    }

#ifdef DEBUG
    reportResult(cars, 0);
#endif

    int num_threads = std::min(N, omp_get_max_threads());
    int chunk_size = (N + num_threads - 1) / num_threads;

    std::vector<int> st(num_threads), ed(num_threads);
    for (int i = 0; i < num_threads; i++) {
        st[i] = i * chunk_size;
        ed[i] = std::min(N, st[i] + chunk_size);
    }

    while (t < T) {
        PRNG base = *traffic_prng::engine;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int start = st[tid], end = ed[tid];

            for (int i = start; i < end; i++) {
                const Car& c = cars[i];
                tmp_cars[i] = c;
                Car& new_c = tmp_cars[i];

                int other_lane = c.lane ^ 1;

                int p2 = find_next_bitmask(lane_bitmask, c.lane, L, c.position);
                int p3 = find_next_bitmask(lane_bitmask, other_lane, L, c.position);
                int p0 = find_prev_bitmask(lane_bitmask, other_lane, L, c.position);

                int d2 = (p2 < 0) ? L : dist(L, c.position, p2);
                int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
                int d0 = (p0 < 0) ? L : dist(L, p0, c.position);

                if (d2 < d3 &&
                    c.v >= d2 &&
                    cur_lanes[other_lane * L + c.position] == -1 &&
                    (p0 < 0 || d0 > cars[cur_lanes[other_lane * L + p0]].v)) {
                    new_c.lane ^= 1;
                }

                int offset = new_c.lane * L + new_c.position;
               
                #pragma omp critical
                {
                    nxt_lanes[offset] = new_c.id;
                    nxt_lane_bitmask[offset / 64] |= (1ULL << (offset % 64));
                }
            }
        }

        cur_lanes.swap(nxt_lanes);
        lane_bitmask.swap(nxt_lane_bitmask);
        cars.swap(tmp_cars);

        std::fill(nxt_lane_bitmask.begin(), nxt_lane_bitmask.end(), 0ULL);
        std::fill(nxt_lanes.begin(), nxt_lanes.end(), -1);

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int start = st[tid], end = ed[tid];

            PRNG e = base;
            e.discard(K + 2 * start);

            for (int i = start; i < end; i++) {
                Car& c = cars[i];
                tmp_cars[i] = c;
                Car& new_c = tmp_cars[i];

                int p2 = find_next_bitmask(lane_bitmask, c.lane, L, c.position);
                int d = (p2 < 0) ? L : dist(L, c.position, p2);
                int v2 = (p2 < 0) ? VMAX : cars[cur_lanes[c.lane * L + p2]].v;

                bool if_acc = false;
                bool ss = flip_coin(P_START, &e);
                bool dec = flip_coin(P_DEC, &e);

                if (c.v == 0 && d > 1) {  // slow start
                    if (force_acc[c.id]) {
                        force_acc[c.id] = 0;
                        new_c.v = 1;
                    } else if (ss) {
                        new_c.v = std::min(d - 1, std::min(c.v + 1, VMAX));
                        if_acc = true;
                    } else {
                        new_c.v = 0;
                        force_acc[c.id] = 1;
                    }
                } else if (!if_acc && !force_acc[c.id]) {
                    if (d <= c.v) {
                        if (c.v < v2 || c.v < 2) {
                            new_c.v = d - 1;
                        } else {
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

                new_c.position = (new_c.position + new_c.v) % L;

                int offset = new_c.lane * L + new_c.position;
                #pragma omp critical
                {
                    nxt_lanes[offset] = new_c.id;
                    nxt_lane_bitmask[offset / 64] |= (1ULL << (offset % 64));
                }
            }
        }

        cur_lanes.swap(nxt_lanes);
        lane_bitmask.swap(nxt_lane_bitmask);
        cars.swap(tmp_cars);

        std::fill(nxt_lane_bitmask.begin(), nxt_lane_bitmask.end(), 0ULL);
        std::fill(nxt_lanes.begin(), nxt_lanes.end(), -1);

        t++;
        K += 2 * N;
    }

    #ifdef DEBUG
    reportFinalResult(cars);
    #endif
}
