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
        config.num_inputs = 2;
        config.num_outputs = 1;
        config.num_hidden = 3;
        config.num_init_conns = 5;

        config.weight_mutate_rate = 0.1;
        config.bias_mutate_rate = 0.075;
        config.enable_mutate_rate = 0.01;
        config.activation_mutate_rate = 0;
        config.node_add_prob = 0.01f;
        config.node_delete_prob = 0.01f;
        config.conn_add_prob = 0.01;
        config.conn_delete_prob = 0.01;

        config.bias_init_mean = 0.0f;
        config.bias_init_stdev = 1.0f;
        config.activation_functions = std::vector<std::function<float(float)>>{
                [](float x) { return 1.0f / (1.0f + std::exp(-x)); }
        };
        config.elitism = 1;
        config.dt = 1.0f;
        config.crossover_config.bias_crossover = blx_alpha { 0.5 };
        config.crossover_config.weight_crossover = blx_alpha { 0.5 };
        config.crossover_config.distance_constant.c1 = 0.5;
        config.crossover_config.distance_constant.c2 = 0.5;
        config.crossover_config.distance_constant.c3 = 0.5;
        config.crossover_config.distance_constant.n = 0.5;

        using network_information = network_information<recurrent>;
        genetic::ga_config<network_information> gconfig;
        configure_neat<recurrent, 0>(config, gconfig);
        gconfig.step = [](const std::vector<genetic::ga_config<network_information>::expression_t>& p) {
            std::vector<float> f;
            for(auto e : p) {
                f.push_back(0.0f);
                auto n = std::get<0>(e);
                n.input(std::vector<float> { 1.0f, 1.0f });
                f.back() += 1.0f - n.get_outputs()[0];
                n.input(std::vector<float> { 1.0f, 0.0f });
                f.back() += n.get_outputs()[0];
                n.input(std::vector<float> { 0.0f, 1.0f });
                f.back() += n.get_outputs()[0];
                n.input(std::vector<float> { 0.0f, 0.0f });
                f.back() += 1.0f - n.get_outputs()[0];
            }
            return f;
        };
        gconfig.callback = [](const std::vector<genetic::ga_config<network_information>::expression_t>& e, const std::vector<float>& f) {
            float average = std::accumulate(f.begin(), f.end(), 0.0f) / f.size();
            float max = *std::max_element(f.begin(), f.end());
            float min = *std::min_element(f.begin(), f.end());
            float size = std::accumulate(e.begin(), e.end(), 0.0f, [](auto&& a, auto&& b) {
                return a + std::get<0>(b).size();
            }) / e.size();
            std::cout << average << " " << max << " " << min << std::endl;
            std::cout << size << std::endl;
        };
        gconfig.population = 20;
        gconfig.epoch = 10000;

        gconfig.fitness_min = 0.0f;
        gconfig.fitness_max = 4.0f;
        genetic::ga<network_information> ga(gconfig);
        ga.run();
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

