//
// Created by hungr on 2020/07/24.
//
#include "neat_genomes.h"

network_information network_information::crossover(const network_information &d1, const network_information &d2) {
    network_information d;
    d.output_num = d1.output_num;
    d.input_num = d1.input_num;
    d.node_num = 0;
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
            if(i < d1.conns.size()) {
                d.conns.push_back(d1.conns[i]);
                i++;
            } else if(j < d2.conns.size()) {
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
