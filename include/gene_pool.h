//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_GENE_POOL_H
#define NEAT_GENE_POOL_H

#include "neat_genomes.h"
#include "graph_algorithm.h"
#include "random_generator.h"

class gene_pool_base {
public:
    gene_pool_base(float bias_init_mean, float bias_init_stdev, std::vector<std::function<float(float)>> actionvation_functions);
    virtual void init_gene(network_information_base& gene, std::uint32_t node_num, std::uint32_t input_num, std::uint32_t output_num);
    virtual void add_node(network_information_base& gene);
    virtual void delete_node(network_information_base& gene);
    virtual void add_connection(network_information_base& gene);
    virtual void delete_connection(network_information_base& gene);
    virtual void mutate_activation(float r, network_information_base& gene);
    virtual void mutate_enable(float r, network_information_base& gene);
    virtual void mutate_bias(float r, network_information_base& gene);
    virtual void mutate_weight(float r, network_information_base& gene);
    virtual ~gene_pool_base();
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

// recurrent genome pool
template <class TNet, class = void>
class gene_pool : public gene_pool_base {
public:
    gene_pool(float bias_init_mean, float bias_init_stdev, const std::vector<std::function<float(float)>>& activation_functions) :
            gene_pool_base(bias_init_mean, bias_init_stdev, activation_functions) {}
};

// feedforward genome pool
template <class TNet>
class gene_pool<TNet, std::void_t<typename TNet::feedforwardable>> : public gene_pool_base {
public:
    gene_pool(float bias_init_mean, float bias_init_stdev, const std::vector<std::function<float(float)>>& activation_functions) :
        gene_pool_base(bias_init_mean, bias_init_stdev, activation_functions) {}
    void add_connection(network_information_base& gene) override {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> conns(gene.conns.size());
        std::transform(gene.conns.begin(), gene.conns.end(), conns.begin(), [](auto&& c) { return std::make_pair(c.in, c.out); });

        std::vector<std::pair<std::uint32_t, std::uint32_t>> conn_pair(gene.node_num * (gene.node_num - 1));
        for(auto i = 0, k = 0; i < gene.node_num; i++) {
            for(auto j = 0; j < gene.node_num; j++) {
                if(i == j) continue;
                conn_pair[k] = std::make_pair(gene.nodes[i].id, gene.nodes[j].id);
                k++;
            }
        }

        auto iter = std::remove_if(conn_pair.begin(), conn_pair.end(), [&conns = gene.conns](auto&& p) {
            return std::find_if(conns.begin(), conns.end(), [&p](auto&& c) -> bool {
                return c.in == p.first && c.out == p.second;
            }) != conns.end();
        });
        auto size = iter - conn_pair.begin();
        if(size == 0) return;
        auto idx = size;
        while(true) {
            idx = random_generator::random<std::size_t>() % size;
            conns.emplace_back(conn_pair[idx].first, conn_pair[idx].second);
            if(is_acyclic(gene.node_num, conns)) break;
            conns.pop_back();
            conn_pair.erase(conn_pair.begin() + idx);
            size--;
            if(size == 0) return;
        }
        auto id = push_gene(conn_pair[idx].first, conn_pair[idx].second);

        auto j = 0;
        for(auto&& c : gene.conns) {
            if(c.id > id) break;
            j++;
        }
        auto weight = random_generator::random_uniform<float>(-1.0, 1.0);
        gene.conns.insert(gene.conns.begin() + j,
                          connection { id, conn_pair[idx].first, conn_pair[idx].second, weight, true });
    }
};

#endif //NEAT_GENE_POOL_H
