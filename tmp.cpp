
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
static inline int find_next(const std::vector<int>& lane, const int L, int position) {
        // min 1 car next
    for (int i = 1; i < L; i++) {
            // mod to wrap back 
        int wrapped_shifted = (position + i) % L;
        if (lane[wrapped_shifted] != -1) {
            return wrapped_shifted;
        } 
    }
    return -1; // not supposed to happen
}
static inline int find_prev(const std::vector<int>& lane, const int L, int position) {
    for (int i = 1; i < L; i++) {
        int wrapped_shifted = (position - i + L) % L;
        if (lane[wrapped_shifted] != -1) {
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

static inline int safe_dist_next(const std::vector<int>& lane, const int L, int position) {
    int p = find_next(lane, L, position);
    return (p < 0) ? L : dist(L, position, p);
}

bool can_switch_lane(const std::vector<Car>& cars, const Car& c, const std::vector<std::vector<int>>& lanes, const int L) {
    int p2 = find_next(lanes[c.lane], L, c.position);
    int p3 = find_next(lanes[c.lane^1], L, c.position);
    int p0 = find_prev(lanes[c.lane^1], L, c.position);
    int d2 = (p2 < 0) ? L : dist(L, c.position, p2);
    int d0 = (p0 < 0) ? L : dist(L, p0, c.position);
    int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
    return d2 < d3 && 
            c.v >= d2 && 
            lanes[c.lane^1][c.position] == -1 
            && (d0 == L || d0 > cars[lanes[c.lane^1][p0]].v);
}

bool decelerate(const std::vector<Car>& cars,const std::vector<int>& lane, const Car& c, const int L, 
        std::vector<int>& speed_snapshot, const int VMAX) {
    int p2 = find_next(lane, L, c.position);
    int d  = (p2 < 0) ? L : dist(L, c.position, p2);
    int v2 = (p2 < 0) ? VMAX : cars[lane[p2]].v; 
    int v1 = c.v;
    if (d <= v1) {
        if (v1 < v2 || v1 < 2) {
            speed_snapshot[c.id] = d - 1;
            return true;
        } else if (v1 >= v2 && v1 >= 2) {
            speed_snapshot[c.id] = std::min(d - 1, v1 - 2);
            return true;
        } else {
            return false;
        }
    }
    if (v1 < d && d <= 2 * v1 && v1 >= v2) {
        speed_snapshot[c.id] = v1 - std::floor((v1 - v2) / 2);
        return true;
    }
    return false;
}

void accelerate(const std::vector<int>& lane, const Car& c, const int L, 
        std::vector<int>& speed_snapshot, const int& VMAX) {
    int p2 = find_next(lane, L, c.position);
    int d = (p2 < 0) ? L : dist(L, c.position, p2);
    int v1 = c.v;
    speed_snapshot[c.id] = std::min(d - 1, std::min(v1 + 1, VMAX));
}

void position_update(std::vector<Car>& cars, std::vector<std::vector<int>>& lanes, const int L, const int N,
        const std::vector<int> speed_snapshot) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
            // compute new position
        int new_position = (cars[i].position + speed_snapshot[i]) % L;
            // update state vectors 
                // lanes
        if (speed_snapshot[i] > 0) {
            lanes[cars[i].lane][cars[i].position] = -1;
            lanes[cars[i].lane][new_position] = i;
        }
            // update car object 
        cars[i].position = new_position;
        cars[i].v = speed_snapshot[i];
    }
}

void executeSimulation(Params params, std::vector<Car> cars) {
    const int N = params.n;
    const int L = params.L;
    const int T = params.steps;
    const int VMAX = params.vmax;
    const double P_START = params.p_start;
    const double P_DEC = params.p_dec;
    int t = 0;
        // force accelerate state 
    std::vector<char> force_acc(N, 0);
        // acceleration decision state
    std::vector<char> accelerating(N, 0);
        // can change lane decision state
    std::vector<char> change_lane(N, 0);
        // lanes and car position states
    std::vector<std::vector<int>> lanes(2, std::vector(L, -1)); 
        // centralized per iteration random decision making
    std::vector<char> ss(N, 0);
    std::vector<char> dec(N, 0);
        // speed snapshot
    std::vector<int> speed_snapshot(N, 0);
    for (auto c : cars) {
        lanes[c.lane][c.position] = c.id;
        speed_snapshot[c.id] = c.v;
    }

    #ifdef DEBUG
    reportResult(cars, 0);
    #endif

    while (t < T) {

        for (int i = 0; i < N; i++) {
            ss[i] = flip_coin(P_START, traffic_prng::engine);
            dec[i] = flip_coin(P_DEC, traffic_prng::engine);
        }
            // reset flags
        std::fill(change_lane.begin(), change_lane.end(), 0);
        std::fill(accelerating.begin(), accelerating.end(), 0);
            // Induce randomness
            // need to skip partition size * idx when omp this to ensure deterministic rng to car correspondence
        PRNG base = *traffic_prng::engine;
        
        #pragma omp parallel 
        {
            // std::uniform_real_distribution<float> local(0.0f, 1.0f)
            // snapshot current engine state
            // #pragma omp for schedule(static)
            // for (int i = 0; i < N; i++) {
            //     PRNG e = base;
            //         // skip used rng
            //     long long k = 2LL * ( (long long)t * N + i );
            //     e.discard(k);
            //     ss[i] = flip_coin(P_START, &e);
            //     dec[i] = flip_coin(P_DEC, &e);
            // }
                // update global engine state
            // #pragma omp single
            // traffic_prng::engine->discard(2LL*N);
            #pragma omp for schedule(static)
                for (int i = 0; i < N; i++) {
                    if (can_switch_lane(cars, cars[i], lanes, L)) {
                        change_lane[i] = 1;
                    }
                }
            #pragma omp barrier
            // apply change lane
            #pragma omp for schedule(static)
                for (int i = 0; i < N; i++) {
                    if (change_lane[i]) {
                        Car& c = cars[i];
                        lanes[c.lane][c.position] = -1;
                        lanes[c.lane^1][c.position] = c.id;
                        c.lane ^= 1;
                    } else {
                        continue;
                    }
                }
            #pragma omp barrier
            // slow start
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                Car& c = cars[i];
                int d2 = safe_dist_next(lanes[c.lane], L, c.position);
                if (c.v == 0 && d2 > 1) {   // satisfy slow start criteria
                    if (force_acc[c.id]) {  // did not accelerate when permitted last round
                        force_acc[c.id] = 0;
                        speed_snapshot[c.id] = 1;
                        accelerating[c.id] = 0;
                        continue;
                    } 
                    if (ss[c.id]) {         // random start
                        accelerate(lanes[c.lane], c, L, speed_snapshot, VMAX);
                        accelerating[c.id] = 1;
                        continue;
                    } else {
                        speed_snapshot[c.id] = 0;
                        force_acc[c.id] = 1;
                        continue;
                    }
                } else {
                    continue;
                }

            }
            #pragma omp barrier
                // deterministic deceleration & acceleration
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                Car& c = cars[i];
                if (accelerating[c.id]) {  // only skip accel path for 1a
                    continue;
                }
                    // allow deceleration even if there was a forced start this step
                if (!decelerate(cars, lanes[c.lane], c, L, speed_snapshot, VMAX)) {
                        // Only accelerate if this car was NOT set by forced start earlier:
                    if (!force_acc[c.id]) { // forced start already assigned v=1
                        accelerate(lanes[c.lane], c, L, speed_snapshot, VMAX);
                    }
                }
            }
            #pragma omp barrier
            // random deceleration
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                Car& c = cars[i];
                if (dec[c.id]) {
                    speed_snapshot[c.id] = std::max(0, speed_snapshot[c.id] - 1);
                }
            }
        }
        position_update(cars, lanes, L, N, speed_snapshot);
        #ifdef DEBUG
            reportResult(cars, t);
        #endif
        t++;
    }
    #ifdef DEBUG
        reportFinalResult(cars);
    #endif
}
