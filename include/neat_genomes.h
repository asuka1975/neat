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

template <class TNet>
struct network_information : network_information_base {
    using expression_t = TNet;
    static network_information<TNet> crossover(const network_information<TNet>& d1, const network_information<TNet>& d2) {
        network_information<TNet> d;
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
                    d.nodes.push_back(d1.nodes[i]);
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
                    d.conns.push_back(d1.conns[i]);
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
