
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
static inline int find_next(std::vector<int> lane, int L, int position) {
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
static inline int find_prev(std::vector<int> lane, int L, int position) {
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

bool can_switch_lane(std::vector<Car>& cars, const Car& c, std::vector<std::vector<int>>& lanes, int L) {
    int c2 = find_next(lanes[c.lane], L, c.position);
    int c3 = find_next(lanes[c.lane^1], L, c.position);
    int c0 = find_prev(lanes[c.lane^1], L, c.position);
    int d2 = dist(L, c2, c.position);
    int d0 = dist(L, c.position, c0);
    int d3 = dist(L, c3, c.position);
    return d2 < d3 && c.v >= d2 && lanes[c.lane^1][c.position] == -1 && d0 > cars[lanes[c.lane^1][c0]].v;
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
    std::vector<char> force_acc(L, 0);
        // can change lane state
    std::vector<char> change_lane(L, 0);
        // lanes and car position states
    std::vector<std::vector<int>> lanes(2, std::vector(L, -1)); 
        // centralized per iteration random decision making
    std::vector<char> ss(N, 0);
    std::vector<char> dec(N, 0);

        // build position state vectors
    for (auto c : cars) {
        lanes[c.lane][c.position] = c.id;
    }

    #ifdef DEBUG
    reportResult(cars, 0);
    #endif

    while (t < T) {
            // Induce randomness
            // need to skip partition size * idx when omp this to ensure deterministic rng to car correspondence
        for (int i = 0; i < N; i++) {
            ss[i] = flip_coin(P_START, traffic_prng::engine);
            dec[i] = flip_coin(P_DEC, traffic_prng::engine);
        }
            // clear change lane decision state
        std::fill(change_lane.begin(), change_lane.end(), 0);
            // update lane change decision state
        for (auto c : cars) {
            if (can_switch_lane(cars, c, lanes, L)) {
                change_lane[c.id] = 1;
            }
        }
            // apply change lane 
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
            // car wise determine start/dec/acc
        for (int i = 0; i < N; i++) {
                // slow start
        }
            // 
    }
}

