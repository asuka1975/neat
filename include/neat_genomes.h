//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_NEAT_GENOMES_H
#define NEAT_NEAT_GENOMES_H

#include <functional>
#include "network.h"
#include "random_generator.h"

struct node {
    std::uint32_t id;
    float bias;
    std::uint32_t activation_function;
};

struct connection {
    std::uint32_t id;
    std::uint32_t in;
    std::uint32_t out;
    float weight;
    bool enable = true;
};

struct network_information_base {
    std::uint32_t node_num;
    std::uint32_t input_num;
    std::uint32_t output_num;
    std::vector<node> nodes;
    std::vector<connection> conns;
    std::vector<std::function<float(float)>> activations;
};

template <int Dig, int Dec> // alpha = 2.24 -> blx_alpha<2, 24>
struct blx_alpha {
    static float crossover(float x, float y) {
        constexpr float a = alpha();
        float min = std::min(x, y);
        float max = std::max(x, y);
        float d = max - min;
        min -= d * a; max += d * a;
        return random_generator::random_uniform<float>(min, max);
    }
    static constexpr float alpha() {
        int i = 1;
        for(; Dec / i; i *= 10) ;
        return Dig + static_cast<float>(Dec) / i;
    }
};

template <class TNet, class TRealCrossover>
struct network_information : network_information_base {
    using expression_t = TNet;
    static network_information<TNet, TRealCrossover> crossover(const network_information<TNet, TRealCrossover>& d1, const network_information<TNet, TRealCrossover>& d2) {
        network_information<TNet, TRealCrossover> d;
        d.output_num = d1.output_num;
        d.input_num = d1.input_num;
        d.node_num = 0;
        d.activations = d1.activations;
        auto input_output = d.input_num + d.output_num;
        d.nodes.reserve(d1.node_num + d2.node_num - input_output);
        for(auto i = 0u, j = 0u; i < d1.nodes.size() || j < d2.nodes.size(); ) {
            if(i < d1.nodes.size() && j < d2.nodes.size()) {
                if(d1.nodes[i].id < d2.nodes[j].id) {
                    d.nodes.push_back(d1.nodes[i]);
                    i++;
                } else if(d1.nodes[i].id > d2.nodes[j].id) {
                    d.nodes.push_back(d2.nodes[j]);
                    j++;
                } else {
                    node n = d1.nodes[i];
                    n.bias = TRealCrossover::crossover(d1.nodes[i].bias, d2.nodes[i].bias);
                    d.nodes.push_back(n);
                    i++, j++;
                }
            } else if(i < d1.nodes.size()) {
                d.nodes.push_back(d1.nodes[i]);
                i++;
            } else if(j < d2.nodes.size()) {
                d.nodes.push_back(d2.nodes[j]);
                j++;
            }
            d.node_num++;
        }
        d.nodes.shrink_to_fit();
        d.conns.reserve(d1.conns.size() + d2.conns.size());
        for (auto i = 0u, j = 0u; i < d1.conns.size() || j < d2.conns.size(); ) {
            if(i < d1.conns.size() && j < d2.conns.size()) {
                if(d1.conns[i].id < d2.conns[j].id) {
                    d.conns.push_back(d1.conns[i]);
                    i++;
                } else if(d1.conns[i].id > d2.conns[j].id) {
                    d.conns.push_back(d2.conns[j]);
                    j++;
                } else {
                    connection c = d1.conns[i];
                    c.weight = TRealCrossover::crossover(d1.conns[i].weight, d2.conns[i].weight);
                    d.conns.push_back(c);
                    i++, j++;
                }
            } else if(i < d1.conns.size()) {
                d.conns.push_back(d1.conns[i]);
                i++;
            } else if(j < d2.conns.size()) {
                d.conns.push_back(d2.conns[j]);
                j++;
            }
        }
        return d;
    }
};

#endif //NEAT_NEAT_GENOMES_H
