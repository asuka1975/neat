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
        config.pool = std::make_shared<gene_pool>(0.0f, 1.0f,
                std::vector<std::function<float(float)>>{
            [](float x) { return 1.0f / (1.0f - std::exp(-x)); }
        });
        config.num_inputs = 2;
        config.num_outputs = 1;
        config.num_hidden = 3;
        config.num_init_conns = 10;

        config.fitness_min = 0.0f;
        config.fitness_max = 4.0f;

        config.weight_mutate_rate = 0.2;
        config.bias_mutate_rate = 0.1;
        config.enable_mutate_rate = 0.02;
        config.activation_mutate_rate = 0;
        config.node_add_prob = 0;
        config.node_delete_prob = 0;
        config.conn_add_prob = 0.05;
        config.conn_delete_prob = 0.05;
        config.step = [](network& n) -> float {
            float f = 0;
            n.input(std::vector<float> { 1.0f, 1.0f });
            f += 1.0f - n.get_outputs()[0];
            n.input(std::vector<float> { 1.0f, 0.0f });
            f += n.get_outputs()[0];
            n.input(std::vector<float> { 0.0f, 1.0f });
            f += n.get_outputs()[0];
            n.input(std::vector<float> { 0.0f, 0.0f });
            f += 1.0f - n.get_outputs()[0];
            return f;
        };
        config.callback = [](const std::vector<genetic::ga<network_information>::individual_t>&, const std::vector<float>& f) {
            float sum = std::accumulate(f.begin(), f.end(), 0.0f) / f.size();
            std::cout << sum << std::endl;
        };
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

