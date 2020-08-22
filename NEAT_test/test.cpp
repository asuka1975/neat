//
// Created by hungr on 2020/07/24.
//
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "neat.h"

namespace {
    TEST(NEAT_BEHAVIOR_TEST, TEST1) {
        neat_config config;
        config.population = 20;
        config.epoch = 1000;
        config.num_inputs = 2;
        config.num_outputs = 1;
        config.num_hidden = 3;
        config.num_init_conns = 5;

        config.fitness_min = 0.0f;
        config.fitness_max = 4.0f;

        config.weight_mutate_rate = 0.1;
        config.bias_mutate_rate = 0.075;
        config.enable_mutate_rate = 0.01;
        config.activation_mutate_rate = 0;
        config.node_add_prob = 0.00f;
        config.node_delete_prob = 0.00f;
        config.conn_add_prob = 0.01;
        config.conn_delete_prob = 0.01;

        config.bias_init_mean = 0.0f;
        config.bias_init_stdev = 1.0f;
        config.is_feedforward = false;
        config.activation_functions = std::vector<std::function<float(float)>>{
                [](float x) { return 1.0f / (1.0f + std::exp(-x)); }
        };

        genetic::ga_config<network_information> gconfig;
        configure_neat<0>(config, gconfig);
        gconfig.step = [](const std::vector<genetic::ga_config<network_information>::expression_t>& p) {
            std::vector<float> f;
            for(auto e : p) {
                f.push_back(0.0f);
                auto n = std::get<0>(e);
                n->input(std::vector<float> { 1.0f, 1.0f });
                f.back() += 1.0f - n->get_outputs()[0];
                n->input(std::vector<float> { 1.0f, 0.0f });
                f.back() += n->get_outputs()[0];
                n->input(std::vector<float> { 0.0f, 1.0f });
                f.back() += n->get_outputs()[0];
                n->input(std::vector<float> { 0.0f, 0.0f });
                f.back() += 1.0f - n->get_outputs()[0];
            }
            return f;
        };
        gconfig.callback = [](const std::vector<genetic::ga_config<network_information>::expression_t>& e, const std::vector<float>& f) {
            float average = std::accumulate(f.begin(), f.end(), 0.0f) / f.size();
            float max = *std::max_element(f.begin(), f.end());
            float min = *std::min_element(f.begin(), f.end());
            float size = std::accumulate(e.begin(), e.end(), 0.0f, [](auto&& a, auto&& b) {
                return a + std::get<0>(b)->size();
            }) / e.size();
            std::cout << average << " " << max << " " << min << std::endl;
            std::cout << size << std::endl;
        };
        genetic::ga<network_information> ga(gconfig);
        ga.run();
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

