//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_NEAT_H
#define NEAT_NEAT_H

#include <functional>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "gene_pool.h"
#include "genetic.h"
#include "network.h"
#include "recurrent.h"
#include "feedforward.h"
#include "devnetwork.h"

struct neat_config {
    std::uint32_t num_hidden;
    std::uint32_t num_inputs;
    std::uint32_t num_outputs;

    std::uint32_t num_init_conns;

    float bias_init_mean = 0.0f;
    float bias_init_stdev = 1.0f;
    std::vector<std::function<float(float)>> activation_functions;
    std::uint32_t elitism = 1;
    float dt = 1.0f;

    float node_add_prob;
    float node_delete_prob;
    float conn_add_prob;
    float conn_delete_prob;
    float enable_mutate_rate;
    float activation_mutate_rate;
    float bias_mutate_rate;
    float weight_mutate_rate;

    std::unique_ptr<gene_pool_base> pool;
};

void to_json(nlohmann::json& j, const neat_config& c);
void from_json(const nlohmann::json& j, neat_config& c);

template <class TNet, std::size_t I, class... TArgs>
void configure_neat(neat_config& config, genetic::ga_config<TArgs...>& gconfig) {
    using network_information_t = typename std::tuple_element<I, std::tuple<TArgs...>>::type;
    using crossover_t = typename network_information_t::real_crossover_t;
    using c1_t = typename network_information_t::c1_t;
    using c2_t = typename network_information_t::c2_t;
    using c3_t = typename network_information_t::c3_t;
    using n_t = typename network_information_t::n_t;
    static_assert(I < std::tuple_size_v<std::tuple<TArgs...>>, "index out of range of parameter pack `TArgs...`");
    static_assert(std::is_same_v<std::tuple_element_t<I, std::tuple<TArgs...>>, network_information<TNet, crossover_t, c1_t, c2_t, c3_t, n_t>>,
            "selected type must be network_information");
    static_assert(std::is_constructible_v<TNet, network_config> &&
            std::is_invocable_v<decltype(&TNet::input), TNet, std::vector<float>> &&
            std::is_invocable_r_v<std::vector<float>, decltype(&TNet::get_outputs), TNet> &&
            std::is_invocable_r_v<std::size_t, decltype(&TNet::size), TNet>,
                    "TNet is not network. TNet must have input(std::vector<float>), get_outputs() and size().");
    using individual_t = typename genetic::ga<TArgs...>::individual_t;
    if(!config.pool) config.pool = std::make_unique<gene_pool<TNet>>(config.bias_init_mean, config.bias_init_stdev, config.activation_functions);

    gconfig.save = config.elitism;
    gconfig.scale = [](float x) { return x * x; };
    gconfig.select = species<TArgs...>{ config.dt, config.elitism };//genetic::elite<TArgs...>{ config.elitism };
    std::get<I>(gconfig.express) = [](const network_information<TNet, crossover_t, c1_t, c2_t, c3_t, n_t>& ni)
            -> TNet {
        network_config config;
        to_network_config(ni, config);
        return TNet(config);
    };
    std::get<I>(gconfig.initializer) = [&pool = config.pool, &config]() {
        network_information<TNet, crossover_t, c1_t, c2_t, c3_t, n_t> n;
        pool->init_gene(n, config.num_inputs + config.num_outputs + config.num_hidden,
                        config.num_inputs, config.num_outputs);
        for(std::size_t i = 0; i < config.num_init_conns; i++) {
            pool->add_connection(n);
        }
        if(std::is_same_v<TNet, devnetwork> && devnet_extensions::enable_evolving_neurocomponents_position) {
            for(node& nd : n.nodes) {
                nd.extra = std::tuple<float, float>(
                        random_generator::random_uniform<float>(-1, 1),
                        random_generator::random_uniform<float>(-1, 1));
            }
        }
        return n;
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
