//
// Created by hungr on 2020/07/24.
//
#include "neat_genomes.h"

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