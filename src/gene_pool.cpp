//
// Created by hungr on 2020/07/24.
//
#include "gene_pool.h"

#include <algorithm>
#include <utility>
#include <set>
#include "graph_algorithm.h"

#include "random_generator.h"

gene_pool_base::gene_pool_base(float bias_init_mean, float bias_init_stdev, std::vector<std::function<float(float)>> activation_functions) :
    bias_init_mean(bias_init_mean), bias_init_stdev(bias_init_stdev), activation_functions(std::move(activation_functions)) {
    node_count = 0;
}

void gene_pool_base::init_gene(network_information_base &gene, std::uint32_t node_num, std::uint32_t input_num,
                          std::uint32_t output_num) {
    gene.node_num = node_num;
    gene.input_num = input_num;
    gene.output_num = output_num;
    node_count = node_num > node_count ? node_num : node_count;
    gene.activations = activation_functions;

    gene.nodes.resize(node_num);
    for(auto i = 0; i < node_num; i++) {
        gene.nodes[i].id = i + 1;
        gene.nodes[i].bias = random_generator::random_normal<float>(bias_init_mean, bias_init_stdev);
        auto j = random_generator::random<std::size_t>() % activation_functions.size();
        gene.nodes[i].activation_function = j;
    }
}

void gene_pool_base::add_node(network_information_base &gene) {
    gene.node_num++;
    node_count++;

    auto idx = random_generator::random<std::size_t>() % activation_functions.size();
    auto bias = random_generator::random_normal(bias_init_mean, bias_init_stdev);
    gene.nodes.push_back( node { node_count, bias, static_cast<uint32_t>(idx)});

    std::vector<std::size_t> indexes;
    for(auto i = 0; i < gene.conns.size(); i++) if(gene.conns[i].enable) indexes.push_back(i);
    auto j = indexes[random_generator::random<std::size_t>() % indexes.size()];
    gene.conns[j].enable = false;

    gene.conns.push_back(connection { push_gene(gene.conns[j].in, node_count), gene.conns[j].in, node_count,
                                      random_generator::random_uniform<float>(-1.0, 1.0 ), true});
    gene.conns.push_back(connection { push_gene(node_count, gene.conns[j].out), node_count, gene.conns[j].out,
                                      random_generator::random_uniform<float>(-1.0, 1.0), true});
}

void gene_pool_base::delete_node(network_information_base &gene) {
    auto hidden_num = gene.node_num - gene.input_num - gene.output_num;
    gene.node_num--;

    auto idx = random_generator::random<std::size_t>() % hidden_num + gene.input_num + gene.output_num;
    auto id = gene.nodes[idx].id;
    gene.nodes.erase(gene.nodes.begin() + idx);

    auto iter = std::remove_if(gene.conns.begin(), gene.conns.end(), [&id](auto&& c) { return c.in == id || c.out == id; });
    gene.conns.erase(iter, gene.conns.end());
}

void gene_pool_base::add_connection(network_information_base &gene) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> conn_pair(gene.node_num * (gene.node_num - 1));
    for(auto i = 0, k = 0; i < gene.node_num; i++) {
        for(auto j = 0; j < gene.node_num; j++) {
            if(i == j) continue;
            conn_pair[k] = std::make_pair(gene.nodes[i].id, gene.nodes[j].id);
            k++;
        }
    }

    auto iter = std::remove_if(conn_pair.begin(), conn_pair.end(), [&conns = gene.conns, &gene](auto&& p) {
        return std::find_if(conns.begin(), conns.end(), [&p, &gene](auto&& c) -> bool {
            return (c.in == p.first && c.out == p.second) || (gene.input_num <= c.in && c.in < gene.input_num + gene.output_num);
        }) != conns.end();
    });

    auto size = iter - conn_pair.begin();
    if(size == 0) return;
    auto idx = random_generator::random<std::size_t>() % size;
    auto id = push_gene(conn_pair[idx].first, conn_pair[idx].second);

    auto j = 0;
    for(auto&& c : gene.conns) {
        if(c.id > id) break;
        j++;
    }
    auto weight = random_generator::random_uniform<float>(-1.0, 1.0);
    gene.conns.insert(gene.conns.begin() + j,
            connection { id, conn_pair[idx].first, conn_pair[idx].second, weight, true });
}

void gene_pool_base::delete_connection(network_information_base &gene) {
    auto idx = random_generator::random<std::size_t>() % gene.conns.size();
    gene.conns.erase(gene.conns.begin() + idx);
}

void gene_pool_base::mutate_activation(float r, network_information_base &gene) {
    for(auto& c : gene.nodes) {
        if(random_generator::random<float>() < r) {
            auto i = random_generator::random<std::size_t>() % activation_functions.size();
            c.activation_function = i;
        }
    }
}

void gene_pool_base::mutate_enable(float r, network_information_base &gene) {
    for(auto& c : gene.conns)
        if(random_generator::random<float>() < r) c.enable = !c.enable;
}

void gene_pool_base::mutate_bias(float r, network_information_base &gene) {
    for(auto& n : gene.nodes)
        if(random_generator::random<float>() < r)
            n.bias = random_generator::random_uniform<float>(-1.0f, 1.0f);
}

void gene_pool_base::mutate_weight(float r, network_information_base &gene) {
    for(auto& c : gene.conns)
        if(random_generator::random<float>() < r)
            c.weight = random_generator::random_uniform<float>(-1.0f, 1.0f);
}

std::uint32_t gene_pool_base::push_gene(std::uint32_t in, std::uint32_t out) {
    auto iter = std::find_if(genes.begin(), genes.end(), [in, out](const connection_gene& gene) -> bool {
        return gene.in == in && gene.out == out;
    });
    if(iter == genes.end()) {
        genes.push_back(connection_gene {static_cast<std::uint32_t>(genes.size() + 1), in, out });
        return genes.size();
    }
    else {
        return iter->id;
    }
}

gene_pool_base::~gene_pool_base() = default;
