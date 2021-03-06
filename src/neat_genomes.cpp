//
// Created by hungr on 2020/07/24.
//
#include "neat_genomes.h"

std::function<float(float, float)> create_real_crossover(const std::string& crossover_str, const nlohmann::json& param) {
    if(crossover_str == "blx-alpha") {
        return blx_alpha(param.get<float>());
    }
    return std::function<float(float, float)>{};
}

using json = nlohmann::json;

void to_json(nlohmann::json& j, const node& n) {
    j = {
            { "id", n.id },
            { "bias", n.bias },
            { "activation_function", n.activation_function }
    };
}

void from_json(const nlohmann::json& j, node& n) {
    n.id = j.at("id").get<std::uint32_t>();
    n.bias = j.at("bias").get<float>();
    n.activation_function = j.at("activation_function").get<std::uint32_t>();
}

void to_json(nlohmann::json& j, const connection& n) {
    j = {
            { "id", n.id },
            { "in", n.in },
            { "out", n.out },
            { "weight", n.weight },
            { "enable", n.enable }
    };
}

void from_json(const nlohmann::json& j, connection& n) {
    n.id = j.at("id").get<std::uint32_t>();
    n.in = j.at("in").get<std::uint32_t>();
    n.out = j.at("out").get<std::uint32_t>();
    n.weight = j.at("weight").get<float>();
    n.enable = j.at("enable").get<bool>();
}

void to_json(nlohmann::json& j, const network_information_base& n) {
    j = {
            { "input_num", n.input_num },
            { "output_num", n.output_num },
            { "node_num", n.node_num },
            { "nodes", n.nodes },
            { "conns", n.conns }
    };
}

void from_json(const nlohmann::json& j, network_information_base& n) {
    n.input_num = j.at("input_num").get<std::uint32_t>();
    n.output_num = j.at("output_num").get<std::uint32_t>();
    n.node_num = j.at("node_num").get<std::uint32_t>();
    n.nodes = j.at("nodes").get<std::vector<node>>();
    n.conns = j.at("conns").get<std::vector<connection>>();
}

void from_json(const nlohmann::json& j, neat_crossover_config& config) {
    config.bias_crossover = create_real_crossover(j.at("bias_crossover")[0].get<std::string>(), j.at("bias_crossover")[1]);
    config.weight_crossover = create_real_crossover(j.at("weight_crossover")[0].get<std::string>(), j.at("weight_crossover")[1]);
    auto& dc = j.at("distance_constant");
    config.distance_constant.c1 = dc.at("c1").get<float>();
    config.distance_constant.c2 = dc.at("c2").get<float>();
    config.distance_constant.c3 = dc.at("c3").get<float>();
    config.distance_constant.n = dc.at("n").get<float>();
}

void to_network_config(const network_information_base& ni, network_config& config) {
    config.input_num = ni.input_num;
    config.output_num = ni.output_num;
    config.conn.resize(ni.conns.size());
    config.node.resize(ni.nodes.size());
    std::map<std::uint32_t, std::uint32_t> id_to_index;
    for(std::size_t i = 0; i < ni.nodes.size(); i++) id_to_index[ni.nodes[i].id] = i;
    std::transform(ni.conns.begin(), ni.conns.end(), config.conn.begin(),
                   [&id_to_index](const auto& c) {
                       return c.enable ?
                              conn_t { id_to_index[c.in], id_to_index[c.out], c.weight, nullptr } :
                              conn_t { 0u, 0u, c.weight, nullptr }; });
    auto iter = std::remove_if(config.conn.begin(), config.conn.end(),
                               [](auto&& c) { return c.in == 0 && c.out == 0; });
    config.conn.erase(iter, config.conn.end());
    std::transform(ni.nodes.begin(), ni.nodes.end(), config.node.begin(),
                   [](auto&& n) { return node_t { n.activation_function, n.bias, nullptr }; });
    config.f = ni.activations;
}

blx_alpha::blx_alpha(float a) : alpha(a) {}

float blx_alpha::operator()(float x, float y) const {
    float min = std::min(x, y);
    float max = std::max(x, y);
    float d = max - min;
    min -= d * alpha; max += d * alpha;
    return random_generator::random_uniform<float>(min, max);
}
