//
// Created by hungr on 2020/07/24.
//
#include "neat.h"

namespace {
    genetic::ga_config<network_information> create_ga_config(const neat_config& config);
}

neat::neat(const neat_config &config) : algo(create_ga_config(config)), config(config) {

}

namespace {
    genetic::ga_config<network_information> create_ga_config(const neat_config& config) {
        genetic::ga_config<network_information> gconfig;
        gconfig.population = config.population;
        gconfig.epoch = config.epoch;
        gconfig.fitness_max = config.fitness_max;
        gconfig.fitness_min = config.fitness_min;
        gconfig.scale = [](float x) { return x * x; };
        gconfig.select = genetic::elite<network_information>{ 3 };
        gconfig.step = config.step;
        gconfig.callback = config.callback;
        std::weak_ptr<gene_pool> weak = config.pool;
        using individual_t = genetic::ga<network_information>::individual_t;
        gconfig.initializer = [weak, config]() -> individual_t {
            auto d = std::make_tuple(network_information{});
            auto& n = std::get<0>(d);
            if(auto pool = weak.lock()) {
                pool->init_gene(n, config.num_inputs + config.num_outputs + config.num_hidden,
                        config.num_inputs, config.num_outputs);
                for(auto i = 0; i < config.num_init_conns; i++) {
                    pool->add_connection(n);
                }
            }
            return d;
        };
        gconfig.mutates.emplace_back(config.node_add_prob, [weak](individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->add_node(std::get<0>(d));
        });
        gconfig.mutates.emplace_back(config.node_delete_prob, [weak](individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->delete_node(std::get<0>(d));
        });
        gconfig.mutates.emplace_back(config.conn_add_prob, [weak](individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->add_connection(std::get<0>(d));
        });
        gconfig.mutates.emplace_back(config.conn_delete_prob, [weak](individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->delete_connection(std::get<0>(d));
        });
        gconfig.node_mutates.emplace_back(config.enable_mutate_rate, [weak](float r, individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->mutate_enable(r, std::get<0>(d));
        });
        gconfig.node_mutates.emplace_back(config.activation_mutate_rate, [weak](float r, individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->mutate_activation(r, std::get<0>(d));
        });
        gconfig.node_mutates.emplace_back(config.bias_mutate_rate, [weak](float r, individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->mutate_bias(r, std::get<0>(d));
        });
        gconfig.node_mutates.emplace_back(config.weight_mutate_rate, [weak](float  r, individual_t& d) -> void {
            if(auto pool = weak.lock()) pool->mutate_weight(r, std::get<0>(d));
        });
        return gconfig;
    }
}
