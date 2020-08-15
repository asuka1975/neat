//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_NEAT_GENOMES_H
#define NEAT_NEAT_GENOMES_H

#include <functional>
#include "network.h"

struct node {
    std::uint32_t id;
    float bias;
    std::function<float(float)> activation_function;
};

struct connection {
    std::uint32_t id;
    std::uint32_t in;
    std::uint32_t out;
    float weight;
    bool enable = true;
};

struct network_information {
    using expression_t = network;
    std::uint32_t node_num;
    std::uint32_t input_num;
    std::uint32_t output_num;
    std::vector<node> nodes;
    std::vector<connection> conns;
    static network_information crossover(const network_information& d1, const network_information& d2);
};

namespace genetic {
    network_information crossover(const network_information& d1, const network_information& d2);
}

#endif //NEAT_NEAT_GENOMES_H
