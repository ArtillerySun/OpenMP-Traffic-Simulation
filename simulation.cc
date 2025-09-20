
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

void executeSimulation(Params params, std::vector<Car> cars) {
  int n = params.n;
  int timesteps = params.steps;
  int t = 0;
  int vmax = params.vmax;
  double p_start = params.p_start;
  double p_dec = params.p_dec;
  while (t < timesteps) {
    
    for (int i = 0; i < n; i++) {

    }
  }
}

int c2_id(const std::vector<Car>& cars, const Car& c_, int N) {
    int best_id = -1;
    int best_dist = N + 1;

    for (auto const& c : cars) {
        if (c.lane != c_.lane || c.id == c_.id) continue;

        int dist = (c.position - c_.position + N) % N;
        if (dist > 0 && dist < best_dist) {
            best_dist = dist;
            best_id = c.id;
        }
    }
    return best_id;
}

int c0_id(const std::vector<Car>& cars, const Car& c_, int N) {
    int best_id = -1;
    int best_dist = N + 1;

    for (auto const& c : cars) {
        if (c.lane == c_.lane || c.id == c_.id) continue;

        int dist = (c_.position - c.position + N) % N;
        if (dist > 0 && dist < best_dist) {
            best_dist = dist;
            best_id = c.id;
        }
    }
    return best_id;
}


int c3_id(const std::vector<Car>& cars, const Car& c_, int N) {
    int best_id = -1;
    int best_dist = N + 1;

    for (auto const& c : cars) {
        if (c.lane == c_.lane || c.id == c_.id) continue;

        int dist = (c.position - c_.position + N) % N;
        if (dist > 0 && dist < best_dist) {
            best_dist = dist;
            best_id = c.id;
        }
    }
    return best_id;
}

bool vacant_next_l(std::vector<Car>& cars, const Car& c_, int N) {
    for (auto const& c : cars) {
        if (c.lane == c_.lane) continue;
        if (c.position == c_.position) return false;
    }
    return true;
}

bool can_switch(std::vector<Car>& cars, const Car& c_, int N) {
    auto c0 = cars[c0_id(cars, c_, N)];
    auto c2 = cars[c2_id(cars, c_, N)];
    auto c3 = cars[c3_id(cars, c_, N)];
    int d0 = c_.position - c0.position;
    int d2 = c2.position - c_.position;
    int d3 = c3.position - c_.position;
    return d2 < d3 && c_.v >= d2 && d0 > c0.v && vacant_next_l(cars, c_, N);
}

bool is_moving(const Car& c) {
    return c.v != 0;
}

bool can_move(std::vector<Car>& cars, const Car& c_, int N) {
    auto c2 = cars[c2_id(cars, c_, N)];
    int d2 = c2.position - c_.position;
    return c_.v == 0 && d2 > 1;
}

bool acc_p(std::vector<Car>& cars, Car c_, int N, double ss, const double& p, const int& vmax) {
    if (ss == 1.0 || ss >= p) {
        auto c2 = cars[c2_id(cars, c_, N)];
        int d2 = c2.position - c_.position;
        c_.v = std::min(d2 - 1, std::min(c_.v + 1, vmax));
        return true;
    }
    return false;
}

bool deterministic_dec(std::vector<Car>& cars, Car c_, int N) {
    auto c2 = cars[c2_id(cars, c_, N)];
    int d2 = c2.position - c_.position;
    if (d2 <= c_.v && (c_.v < c2.v || c_.v < 2)) {
        c_.v = d2 - 1;
        return true;
    }
    if (d2 <= c_.v && c_.v >= c2.v && c_.v >= 2) {
        c_.v = std::min(d2 - 1, c_.v - 2);
        return true;
    }
    if (c_.v < d2 && d2 <= 2 * c_.v && c_.v >= c2.v) {
        c_.v = c_.v - std::floor((c_.v - c2.v)/2.0);
        return true;
    }
    return false;
}

void deterministic_dec_acc(std::vector<Car>& cars, Car c_, int N) {
    if (!deterministic_dec) {

    }
}


void single_update(std::vector<Car>& cars, Car c, int N, const int& vmax,
    const double& p_start, double ss, 
    const double& p_dec, double dec) {
    
    if (can_move(cars, c, N)) {
        
    }
}