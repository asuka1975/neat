//
// Created by hungr on 2020/07/24.
//
#include "neat.h"


void to_json(nlohmann::json& j, const neat_config& c) {
    j = {
            { "num_inputs", c.num_inputs },
            { "num_outputs", c.num_outputs },
            { "num_hidden", c.num_hidden },
            { "num_init_conns", c.num_init_conns },
            { "bias_init_stdev", c.bias_init_stdev },
            { "bias_init_mean", c.bias_init_mean },
            { "bias_mutate_rate", c.bias_mutate_rate },
            { "activation_mutate_rate", c.activation_mutate_rate },
            { "enable_mutate_rate", c.enable_mutate_rate },
            { "weight_mutate_rate", c.weight_mutate_rate },
            { "conn_add_prob", c.conn_add_prob },
            { "conn_delete_prob", c.conn_delete_prob },
            { "node_add_prob", c.node_add_prob },
            { "node_delete_prob", c.node_delete_prob },
            { "gene_pool", c.pool->to_json() }
    };
}

void from_json(const nlohmann::json& j, neat_config& c) {
    c.num_inputs = j.at("num_inputs").get<std::uint32_t>();
    c.num_outputs = j.at("num_outputs").get<std::uint32_t>();
    c.num_hidden = j.at("num_hidden").get<std::uint32_t>();
    c.num_init_conns = j.at("num_init_conns").get<std::uint32_t>();
    c.bias_init_stdev = j.at("bias_init_stdev").get<float>();
    c.bias_init_mean = j.at("bias_init_mean").get<float>();
    c.bias_mutate_rate = j.at("bias_mutate_rate").get<float>();
    c.activation_mutate_rate = j.at("activation_mutate_rate").get<float>();
    c.enable_mutate_rate = j.at("enable_mutate_rate").get<float>();
    c.weight_mutate_rate = j.at("weight_mutate_rate").get<float>();
    c.conn_add_prob = j.at("conn_add_prob").get<float>();
    c.conn_delete_prob = j.at("conn_delete_prob").get<float>();
    c.node_add_prob = j.at("node_add_prob").get<float>();
    c.node_delete_prob = j.at("node_delete_prob").get<float>();
    c.pool->from_json(j.at("gene_pool"));
}