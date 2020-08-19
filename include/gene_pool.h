//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_GENE_POOL_H
#define NEAT_GENE_POOL_H

#include "neat_genomes.h"

class gene_pool {
public:
    gene_pool(float bias_init_mean, float bias_init_stdev, std::vector<std::function<float(float)>> actionvation_functions);
    virtual void init_gene(network_information& gene, std::uint32_t node_num, std::uint32_t input_num, std::uint32_t output_num);
    virtual void add_node(network_information& gene);
    virtual void delete_node(network_information& gene);
    virtual void add_connection(network_information& gene);
    virtual void delete_connection(network_information& gene);
    virtual void mutate_activation(float r, network_information& gene);
    virtual void mutate_enable(float r, network_information& gene);
    virtual void mutate_bias(float r, network_information& gene);
    virtual void mutate_weight(float r, network_information& gene);
    virtual ~gene_pool();
protected:
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

class feedforward_gene_pool : public gene_pool {
public:
    feedforward_gene_pool(float bias_init_mean, float bias_init_stdev, std::vector<std::function<float(float)>> actionvation_functions);
    void init_gene(network_information& gene, std::uint32_t node_num, std::uint32_t input_num, std::uint32_t output_num) override;
    void add_node(network_information& gene) override;
    void delete_node(network_information& gene) override;
    void add_connection(network_information& gene) override;
    void delete_connection(network_information& gene) override;
    void mutate_activation(float r, network_information& gene) override;
    void mutate_enable(float r, network_information& gene) override;
    void mutate_bias(float r, network_information& gene) override;
    void mutate_weight(float r, network_information& gene) override;
};

class recurrent_gene_pool : public gene_pool {

};

#endif //NEAT_GENE_POOL_H
