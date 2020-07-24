//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_NEAT_H
#define NEAT_NEAT_H

#include <functional>
#include <memory>
#include <vector>
#include "gene_pool.h"
#include "genetic.h"
#include "network.h"

struct neat_config {
    std::uint32_t num_hidden;
    std::uint32_t num_inputs;
    std::uint32_t num_outputs;

    std::uint32_t num_init_conns;

    float bias_init_mean = 0.0f;
    float bias_init_stdev = 1.0f;
    std::vector<std::function<float(float)>> activation_functions;

    float node_add_prob;
    float node_delete_prob;
    float conn_add_prob;
    float conn_delete_prob;
    float enable_mutate_rate;
    float activation_mutate_rate;
    float bias_mutate_rate;
    float weight_mutate_rate;

    float fitness_max;
    float fitness_min;

    std::uint32_t population;
    std::uint32_t epoch;

    std::shared_ptr<gene_pool> pool;
    std::function<float(network&)> step;
    std::function<void(const std::vector<genetic::ga<network_information>::individual_t>&, const std::vector<float>&)> callback;
};

class neat {
public:
    explicit neat(const neat_config& config);
    void run();
private:
    neat_config config;
    genetic::ga<network_information> algo;
};

#endif //NEAT_NEAT_H
