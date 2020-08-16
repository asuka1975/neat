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
    std::uint32_t elitism = 1;

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

    std::unique_ptr<gene_pool> pool;
};

template <std::size_t I, class... TArgs>
void configure_neat(neat_config& config, genetic::ga_config<TArgs...>& gconfig) {
    static_assert(I < std::tuple_size_v<std::tuple<TArgs...>>, "index out of range of parameter pack `TArgs...`");
    static_assert(std::is_same_v<std::tuple_element_t<I, std::tuple<TArgs...>>, network_information>,
            "selected type must be network_information");
    using individual_t = typename genetic::ga<TArgs...>::individual_t;
    config.pool = std::make_unique<gene_pool>(config.bias_init_mean, config.bias_init_stdev,
            config.activation_functions);
    gconfig.population = config.population;
    gconfig.epoch = config.epoch;
    gconfig.fitness_max = config.fitness_max;
    gconfig.fitness_min = config.fitness_min;
    gconfig.save = config.elitism;
    gconfig.scale = [](float x) { return x * x; };
    gconfig.select = genetic::elite<network_information>{ config.elitism };
    std::get<I>(gconfig.express) = [](const network_information& ni) {
        network_config config;
        config.node_count = ni.node_num;
        config.input_count = ni.input_num;
        config.output_count = ni.output_num;
        config.connection_count = std::count_if(ni.conns.begin(), ni.conns.end(),
                                                [](const auto& c) { return c.enable; });
        config.bias.resize(ni.nodes.size());
        config.activations.resize(ni.nodes.size());
        config.weight.resize(ni.conns.size());
        std::transform(ni.nodes.begin(), ni.nodes.end(), config.bias.begin(),
                       [](const auto& n) { return n.bias; });
        std::transform(ni.nodes.begin(), ni.nodes.end(), config.activations.begin(),
                       [](const auto& n) { return n.activation_function; });
        std::transform(ni.conns.begin(), ni.conns.end(), config.weight.begin(),
                       [](const auto& c) { return c.weight; });
        std::map<std::uint32_t, std::uint32_t> id_to_index;
        for(auto i = 0; i < ni.nodes.size(); i++) id_to_index[ni.nodes[i].id] = i;

        config.connection_rule.resize(ni.conns.size());
        std::transform(ni.conns.begin(), ni.conns.end(), config.connection_rule.begin(),
                       [&id_to_index](const auto& c) {
                           return c.enable ? std::make_pair(id_to_index[c.in], id_to_index[c.out]) : std::make_pair(0u, 0u);
                       });
        auto iter = std::remove_if(config.connection_rule.begin(), config.connection_rule.end(),
                                   [](auto&& p) { return p.first == 0 && p.second == 0; });
        config.connection_rule.erase(iter, config.connection_rule.end());
        return network(config);
    };
    gconfig.initializer = [&pool = config.pool, &config]() -> individual_t {
        std::tuple<TArgs...> d;
        auto& n = std::get<I>(d);
        pool->init_gene(n, config.num_inputs + config.num_outputs + config.num_hidden,
                        config.num_inputs, config.num_outputs);
        for(auto i = 0; i < config.num_init_conns; i++) {
            pool->add_connection(n);
        }
        return d;
    };
    gconfig.mutates.emplace_back(config.node_add_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->add_node(std::get<0>(d));
    });
    gconfig.mutates.emplace_back(config.node_delete_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->delete_node(std::get<0>(d));
    });
    gconfig.mutates.emplace_back(config.conn_add_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->add_connection(std::get<0>(d));
    });
    gconfig.mutates.emplace_back(config.conn_delete_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->delete_connection(std::get<0>(d));
    });
    gconfig.node_mutates.emplace_back(config.enable_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_enable(r, std::get<0>(d));
    });
    gconfig.node_mutates.emplace_back(config.activation_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_activation(r, std::get<0>(d));
    });
    gconfig.node_mutates.emplace_back(config.bias_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_bias(r, std::get<0>(d));
    });
    gconfig.node_mutates.emplace_back(config.weight_mutate_rate, [&pool = config.pool](float  r, individual_t& d) -> void {
        pool->mutate_weight(r, std::get<0>(d));
    });
}

#endif //NEAT_NEAT_H
