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
#include "recurrent.h"
#include "feedforward.h"

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

    bool is_feedforward = true;

    std::unique_ptr<gene_pool> pool;
};

template <std::size_t I, class... TArgs>
void configure_neat(neat_config& config, genetic::ga_config<TArgs...>& gconfig) {
    static_assert(I < std::tuple_size_v<std::tuple<TArgs...>>, "index out of range of parameter pack `TArgs...`");
    static_assert(std::is_same_v<std::tuple_element_t<I, std::tuple<TArgs...>>, network_information>,
            "selected type must be network_information");
    using individual_t = typename genetic::ga<TArgs...>::individual_t;
    if(config.is_feedforward) {
        config.pool = std::make_unique<feedforward_gene_pool>(config.bias_init_mean, config.bias_init_stdev, config.activation_functions);
    } else {
        config.pool = std::make_unique<gene_pool>(config.bias_init_mean, config.bias_init_stdev, config.activation_functions);
    }

    gconfig.population = config.population;
    gconfig.epoch = config.epoch;
    gconfig.fitness_max = config.fitness_max;
    gconfig.fitness_min = config.fitness_min;
    gconfig.save = config.elitism;
    gconfig.scale = [](float x) { return x * x; };
    gconfig.select = genetic::elite<network_information>{ config.elitism };
    std::get<I>(gconfig.express) = [is_feedforward=config.is_feedforward](const network_information& ni)
            -> std::shared_ptr<network> {
        network_config config;
        config.input_num = ni.input_num;
        config.output_num = ni.output_num;
        config.conn.resize(ni.conns.size());
        config.node.resize(ni.nodes.size());
        std::map<std::uint32_t, std::uint32_t> id_to_index;
        for(auto i = 0; i < ni.nodes.size(); i++) id_to_index[ni.nodes[i].id] = i;
        std::transform(ni.conns.begin(), ni.conns.end(), config.conn.begin(),
                [&id_to_index](const auto& c) {
            return c.enable ?
                std::make_tuple(id_to_index[c.in], id_to_index[c.out], c.weight) :
                std::make_tuple(0u, 0u, c.weight); });
        auto iter = std::remove_if(config.conn.begin(), config.conn.end(),
                [](auto&& c) { return std::get<0>(c) == 0 && std::get<1>(c) == 0; });
        config.conn.erase(iter, config.conn.end());
        std::transform(ni.nodes.begin(), ni.nodes.end(), config.node.begin(),
                [](auto&& n) { return std::make_tuple(n.activation_function, n.bias); });
        config.f = ni.activations;
        if(is_feedforward) {
            return std::make_shared<feedforward>(config);
        } else {
            return std::make_shared<recurrent>(config);
        }
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
        pool->add_node(std::get<I>(d));
    });
    gconfig.mutates.emplace_back(config.node_delete_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->delete_node(std::get<I>(d));
    });
    gconfig.mutates.emplace_back(config.conn_add_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->add_connection(std::get<I>(d));
    });
    gconfig.mutates.emplace_back(config.conn_delete_prob, [&pool = config.pool](individual_t& d) -> void {
        pool->delete_connection(std::get<I>(d));
    });
    gconfig.node_mutates.emplace_back(config.enable_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_enable(r, std::get<I>(d));
    });
    gconfig.node_mutates.emplace_back(config.activation_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_activation(r, std::get<I>(d));
    });
    gconfig.node_mutates.emplace_back(config.bias_mutate_rate, [&pool = config.pool](float r, individual_t& d) -> void {
        pool->mutate_bias(r, std::get<I>(d));
    });
    gconfig.node_mutates.emplace_back(config.weight_mutate_rate, [&pool = config.pool](float  r, individual_t& d) -> void {
        pool->mutate_weight(r, std::get<I>(d));
    });
}

#endif //NEAT_NEAT_H
