
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
    // on the given lane, find the position of the immediately next car
static inline int find_next(const std::set<int>& positions, int position) {
    auto it = positions.upper_bound(position);
    
    if (it != positions.end()) {
        return *it;
    }

    if (positions.size() > 1) {
        return *positions.begin();
    }

    return -1;
}
static inline int find_prev(const std::set<int>& positions, int position) {
    auto it = positions.lower_bound(position);
    
    if (it != positions.begin()) {
        return *(--it);
    }

    if (positions.size() > 1) {
        return *positions.rbegin();
    }

    return -1;
}
    // compute distance from 2 position on circular road
static inline int dist(int L, int prev, int next) { 
    int d = (next - prev + L) % L; 
    return d == 0 ? L : d; 
}

static inline int safe_dist_next(const std::set<int>& lane, const int L, int position) {
    int p = find_next(lane, position);
    return (p < 0) ? L : dist(L, position, p);
}

bool can_switch_lane(const Car& c, const std::vector<std::set<int>>& lanes, 
    std::vector<std::unordered_map<int, Car*> >& mps, const int L) {
    int p2 = find_next(lanes[c.lane], c.position);
    int p3 = find_next(lanes[c.lane^1], c.position);
    int p0 = find_prev(lanes[c.lane^1], c.position);
    int d2 = (p2 < 0) ? L : dist(L, c.position, p2);
    int d0 = (p0 < 0) ? L : dist(L, p0, c.position);
    int d3 = (p3 < 0) ? L : dist(L, c.position, p3);
    return d2 < d3 && 
            c.v >= d2 && 
            mps[c.lane^1].find(c.position) == mps[c.lane^1].end()  
            && (d0 == L || d0 > mps[c.lane^1][p0]->v);
}

bool decelerate(const std::set<int>& lane, 
    std::unordered_map<int, Car*>& mp, const Car& c, const int L, 
        std::vector<int>& speed_snapshot, const int VMAX) {

    
    int p2 = find_next(lane, c.position);
    // std::cerr << c.position << ' ' << p2 << std::endl;
    int d  = (p2 < 0) ? L : dist(L, c.position, p2);
    int v2 = (p2 < 0) ? VMAX : mp[p2]->v; 
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
        speed_snapshot[c.id] = v1 - (v1 - v2) / 2;
        return true;
    }
    return false;
}

void accelerate(const std::set<int>& lane, const Car& c, const int L, 
        std::vector<int>& speed_snapshot, const int& VMAX) {
    int p2 = find_next(lane, c.position);
    int d = (p2 < 0) ? L : dist(L, c.position, p2);
    int v1 = c.v;
    speed_snapshot[c.id] = std::min(d - 1, std::min(v1 + 1, VMAX));
}

void position_update(std::vector<Car>& cars, std::vector<std::set<int>>& lanes, 
    std::vector<std::unordered_map<int, Car*> >& mps, const int L, const int N, const std::vector<int> speed_snapshot) {
    for (int i = 0; i < N; i++) {
            // compute new position
        Car* c = &cars[i];
        int new_position = (c->position + speed_snapshot[i]) % L;

        if (speed_snapshot[i] > 0) {
            auto it = lanes[c->lane].lower_bound(c->position);
            if (it != lanes[c->lane].end()) lanes[c->lane].erase(it);
            mps[c->lane].erase(c->position);
            
            lanes[c->lane].insert(new_position);
            mps[c->lane][new_position] = c;
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
        // centralized per iteration random decision making
    std::vector<std::set<int>> lane_positions(2);
    std::vector<std::unordered_map<int, Car*>> mps(2);

    std::vector<char> ss(N, 0);
    std::vector<char> dec(N, 0);
        // speed snapshot
    std::vector<int> speed_snapshot(N, 0);
        // build position and speed state vectors
    lane_positions[0].clear();
    lane_positions[1].clear();

    for (auto &c : cars) lane_positions[c.lane].insert(c.position);
    for (auto &c : cars) mps[c.lane][c.position] = &c;
    for (auto &c : cars) speed_snapshot[c.id] = c.v;

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
            // clear accelerating and change lane decision state
        std::fill(change_lane.begin(), change_lane.end(), 0);
        std::fill(accelerating.begin(), accelerating.end(), 0);
            // update lane change decision state
        for (auto c : cars) {
            if (can_switch_lane(c, lane_positions, mps, L)) {
                change_lane[c.id] = 1;
            }
        }
            // apply change lane 
        for (int i = 0; i < N; i++) {
            if (change_lane[i]) {
                Car* c = &cars[i];
                // lanes[c.lane][c.position] = -1;
                // lanes[c.lane^1][c.position] = c.id;
                
                auto it = lane_positions[c->lane].find(c->position);
                if(it != lane_positions[c->lane].end()) lane_positions[c->lane].erase(it);
                mps[c->lane].erase(c->position);
                
                c->lane ^= 1;
                lane_positions[c->lane].insert(c->position);
                mps[c->lane][c->position] = c;

            } else {
                continue;
            }
        }

        // #ifdef DEBUG
        // reportResult(cars, t);
        // #endif
            // car wise determine start/dec/acc
            // slow start
        for (auto c : cars) {
            int d2 = safe_dist_next(lane_positions[c.lane], L, c.position);
            if (c.v == 0 && d2 > 1) {   // satisfy slow start criteria
                if (force_acc[c.id]) {  // did not accelerate when permitted last round
                    force_acc[c.id] = 0;
                    speed_snapshot[c.id] = 1;
                    accelerating[c.id] = 0;
                    continue;
                } 
                if (ss[c.id]) {         // random start
                    accelerate(lane_positions[c.lane], c, L, speed_snapshot, VMAX);
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
        
            // deterministic deceleration & acceleration
        for (auto &c : cars) {
            
            if (accelerating[c.id]) {  // only skip accel path for 1a
                continue;
            }
                // allow deceleration even if there was a forced start this step
            if (!decelerate(lane_positions[c.lane], mps[c.lane], c, L, speed_snapshot, VMAX)) {
                // std::cerr<<'#'<<c.id<<std::endl;
                    // Only accelerate if this car was NOT set by forced start earlier:
                if (!force_acc[c.id]) { // forced start already assigned v=1
                    accelerate(lane_positions[c.lane], c, L, speed_snapshot, VMAX);
                }
            }
        }
            // random deceleration
        for (auto c : cars) {
            if (dec[c.id]) {
                speed_snapshot[c.id] = std::max(0, speed_snapshot[c.id] - 1);
            }
        }

        
        position_update(cars, lane_positions, mps, L, N, speed_snapshot);
        #ifdef DEBUG
            reportResult(cars, t);
        #endif
        t++;
    }
    #ifdef DEBUG
        reportFinalResult(cars);
    #endif
}

