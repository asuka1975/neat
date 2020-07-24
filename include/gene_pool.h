//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_GENE_POOL_H
#define NEAT_GENE_POOL_H

#include "neat_genomes.h"

class gene_pool {
public:
    gene_pool(float bias_init_mean, float bias_init_stdev, std::vector<std::function<float(float)>> actionvation_functions);
    void init_gene(network_information& gene, std::uint32_t node_num, std::uint32_t input_num, std::uint32_t output_num);
    void add_node(network_information& gene);
    void delete_node(network_information& gene);
    void add_connection(network_information& gene);
    void delete_connection(network_information& gene);
    void mutate_activation(float r, network_information& gene);
    void mutate_enable(float r, network_information& gene);
    void mutate_bias(float r, network_information& gene);
    void mutate_weight(float r, network_information& gene);
private:
    float bias_init_mean;
    float bias_init_stdev;
    std::vector<std::function<float(float)>> activation_functions;
    std::uint32_t push_gene(std::uint32_t in, std::uint32_t out);
    struct connection_gene {
        std::uint32_t id;
        std::uint32_t in;
        std::uint32_t out;
    };
    std::vector<connection_gene> genes;
    std::uint32_t node_count;
};

#endif //NEAT_GENE_POOL_H
