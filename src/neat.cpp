//
// Created by hungr on 2020/07/24.
//
#include "neat.h"

namespace {
    genetic::ga_config<network_information> create_ga_config(const neat_config& config);
}

neat::neat(const neat_config &config) : algo(create_ga_config(config)), config(config) {

}

void neat::run() {
    algo.run();
}

namespace {
    genetic::ga_config<network_information> create_ga_config(const neat_config& config) {
        using individual_t = genetic::ga<network_information>::individual_t;
        genetic::ga_config<network_information> gconfig;
        gconfig.population = config.population;
        gconfig.epoch = config.epoch;
        gconfig.fitness_max = config.fitness_max;
        gconfig.fitness_min = config.fitness_min;
        gconfig.scale = [](float x) { return x * x; };
        gconfig.select = genetic::elite<network_information>{ 3 };
        gconfig.step = [step = config.step](const std::vector<individual_t>& d) -> std::vector<float> {
            std::vector<float> f;
            for(const auto& i : d) {
                const auto& ni = std::get<0>(i);
                network_config config;
                config.node_count = ni.node_num;
                config.input_count = ni.input_num;
                config.output_count = ni.output_num;
                config.connection_count = std::count_if(ni.conns.begin(), ni.conns.end(),
                        [](const auto& c) { return c.enable; });
                std::transform(ni.nodes.begin(), ni.nodes.end(), std::back_inserter(config.bias),
                        [](const auto& n) { return n.bias; });
                std::transform(ni.nodes.begin(), ni.nodes.end(), std::back_inserter(config.activations),
                               [](const auto& n) { return n.activation_function; });
                std::transform(ni.conns.begin(), ni.conns.end(), std::back_inserter(config.weight),
                        [](const auto& c) { return c.weight; });

                std::transform(ni.conns.begin(), ni.conns.end(), std::back_inserter(config.connection_rule),
                        [](const auto& c) { return c.enable ? std::make_pair(c.in - 1, c.out - 1) : std::make_pair(0u, 0u); });
                auto iter = std::remove_if(config.connection_rule.begin(), config.connection_rule.end(),
                        [](auto&& p) { return p.first == 0 && p.second == 0; });
                config.connection_rule.erase(iter, config.connection_rule.end());
                network n(config);
                f.push_back(step(n));
            }
            return f;
        };
        gconfig.callback = config.callback;
        std::weak_ptr<gene_pool> weak = config.pool;
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
