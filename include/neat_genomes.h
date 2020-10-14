//
// Created by hungr on 2020/07/24.
//

#ifndef NEAT_NEAT_GENOMES_H
#define NEAT_NEAT_GENOMES_H

#include <functional>
#include <list>
#include "network.h"
#include "random_generator.h"
#include "ga.h"
#include "selector.h"

struct node {
    std::uint32_t id;
    float bias;
    std::uint32_t activation_function;
};

struct connection {
    std::uint32_t id;
    std::uint32_t in;
    std::uint32_t out;
    float weight;
    bool enable = true;
};

struct network_information_base {
    std::uint32_t node_num;
    std::uint32_t input_num;
    std::uint32_t output_num;
    std::vector<node> nodes;
    std::vector<connection> conns;
    std::vector<std::function<float(float)>> activations;
};

template <int Dig, int Dec> // alpha = 2.24 -> blx_alpha<2, 24>
struct blx_alpha {
    static float crossover(float x, float y) {
        constexpr float a = alpha();
        float min = std::min(x, y);
        float max = std::max(x, y);
        float d = max - min;
        min -= d * a; max += d * a;
        return random_generator::random_uniform<float>(min, max);
    }
    static constexpr float alpha() {
        int i = 1;
        for(; Dec / i; i *= 10) ;
        return Dig + static_cast<float>(Dec) / i;
    }
};

template <int Dig, int Dec>
struct literal_float_t {
    static constexpr float value() {
        int i = 1;
        for(; Dec / i; i *= 10) ;
        return Dig + static_cast<float>(Dec) / i;
    }
};

template <class TNet, class TRealCrossover, class C1=literal_float_t<0,0>, class C2=literal_float_t<0,0>, class C3=literal_float_t<0,0>, class N=literal_float_t<1,0>>
struct network_information : network_information_base {
    using expression_t = TNet;
    using real_crossover_t = TRealCrossover;
    using c1_t = C1;
    using c2_t = C2;
    using c3_t = C3;
    using n_t = N;
    static network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t> crossover(const network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t>& d1, const network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t>& d2) {
        network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t> d;
        d.output_num = d1.output_num;
        d.input_num = d1.input_num;
        d.node_num = 0;
        d.activations = d1.activations;
        auto input_output = d.input_num + d.output_num;
        d.nodes.reserve(d1.node_num + d2.node_num - input_output);
        for(auto i = 0u, j = 0u; i < d1.nodes.size() || j < d2.nodes.size(); ) {
            if(i < d1.nodes.size() && j < d2.nodes.size()) {
                if(d1.nodes[i].id < d2.nodes[j].id) {
                    d.nodes.push_back(d1.nodes[i]);
                    i++;
                } else if(d1.nodes[i].id > d2.nodes[j].id) {
                    d.nodes.push_back(d2.nodes[j]);
                    j++;
                } else {
                    node n_ = d1.nodes[i];
                    n_.bias = TRealCrossover::crossover(d1.nodes[i].bias, d2.nodes[i].bias);
                    d.nodes.push_back(n_);
                    i++, j++;
                }
            } else if(i < d1.nodes.size()) {
                d.nodes.push_back(d1.nodes[i]);
                i++;
            } else if(j < d2.nodes.size()) {
                d.nodes.push_back(d2.nodes[j]);
                j++;
            }
            d.node_num++;
        }
        d.nodes.shrink_to_fit();
        d.conns.reserve(d1.conns.size() + d2.conns.size());
        for (auto i = 0u, j = 0u; i < d1.conns.size() || j < d2.conns.size(); ) {
            if(i < d1.conns.size() && j < d2.conns.size()) {
                if(d1.conns[i].id < d2.conns[j].id) {
                    d.conns.push_back(d1.conns[i]);
                    i++;
                } else if(d1.conns[i].id > d2.conns[j].id) {
                    d.conns.push_back(d2.conns[j]);
                    j++;
                } else {
                    connection c = d1.conns[i];
                    c.weight = TRealCrossover::crossover(d1.conns[i].weight, d2.conns[i].weight);
                    d.conns.push_back(c);
                    i++, j++;
                }
            } else if(i < d1.conns.size()) {
                d.conns.push_back(d1.conns[i]);
                i++;
            } else if(j < d2.conns.size()) {
                d.conns.push_back(d2.conns[j]);
                j++;
            }
        }
        return d;
    }
    static float distance(const network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t>& d1, const network_information<TNet, TRealCrossover, c1_t, c2_t, c3_t, n_t>& d2) {
        constexpr float c1 = c1_t::value();
        constexpr float c2 = c2_t::value();
        constexpr float c3 = c3_t::value();
        constexpr float n = n_t::value();
        std::size_t d = 0, e = 0, count = 0;
        float w = 0;
        for(std::size_t i = 0, j = 0; ; ) {
            if(i < d1.conns.size() && j < d2.conns.size()) {
                if(d1.conns[i].id < d2.conns[j].id) {
                    d++;
                    i++;
                } else if(d1.conns[i].id > d2.conns[j].id) {
                    d++;
                    j++;
                } else {
                    i++, j++, count++;
                    w += std::abs(d1.conns[i].weight - d2.conns[j].weight);
                }
            } else if(i < d1.conns.size()) {
                e++;
                i++;
            } else if(j < d2.conns.size()) {
                e++;
                j++;
            } else {
                break;
            }
        }
        return c1 * d / n + c2 * e / n + c3 * w;
    }
};

//selection operator (niching)
template <class... TArgs>
class species {
public:
    explicit species(float dt, std::uint32_t elitism = 0);
    std::vector<typename genetic::ga<TArgs...>::individual_t>
    operator()(const std::vector<typename genetic::ga<TArgs...>::individual_t>& pop, const std::vector<float>& fitness);
private:
    float dt;
    std::uint32_t elitism;
    template <class T> class mapply_impl;
    template <class T, class = void> struct has_distance : std::false_type {};
    template <class T> struct has_distance<T, std::void_t<decltype(T::distance)>> : std::true_type {};
    template <class T, class = void> struct distance_wrapper {
        static float distance(const T&, const T&) { return 0; }
    };
    template <class T> struct distance_wrapper<T, std::enable_if_t<has_distance<T>::value>> {
        static float distance(const T& d1, const T& d2) { return T::distance(d1, d2); }
    };
    template <size_t... I>
    struct mapply_impl<std::index_sequence<I...>> {
        static float mapply(const typename genetic::ga<TArgs...>::individual_t& d1, const typename genetic::ga<TArgs...>::individual_t& d2) {
            static auto distance = [](std::initializer_list<float> args) -> float {
                return std::sqrt(std::accumulate(args.begin(), args.end(), 0, [](float a, float b) { return a + b * b; }));
            };
            return distance({distance_wrapper<typename std::tuple_element<I, std::tuple<TArgs...>>::type>::distance(std::get<I>(d1), std::get<I>(d2))...});
        }
    };
};

template <class... TArgs>
inline species<TArgs...>::species(float dt, std::uint32_t elitism) : dt(dt), elitism(elitism) {}

template <class... TArgs>
inline std::vector<typename genetic::ga<TArgs...>::individual_t>
species<TArgs...>::operator()(const std::vector<typename genetic::ga<TArgs...>::individual_t>& pop, const std::vector<float>& fitness) {
    std::vector<typename genetic::ga<TArgs...>::individual_t> p(pop.size());
    std::vector<float> fitness_updated(fitness.size());
    using index_t = typename std::vector<typename genetic::ga<TArgs...>::individual_t>::size_type;
    std::vector<std::vector<index_t>> specie(pop.size());
    for(auto i = 0; i < pop.size(); i++) {
        float f = 0.0f;
        for(auto j = 0; j < pop.size(); j++) {
            if(mapply_impl<std::index_sequence_for<TArgs...>>::mapply(pop[i], pop[j]) <= dt) {
                f += 1.0f;
                specie[i].push_back(j);
            }
        }
        fitness_updated[i] = fitness[i] / f;
    }
    std::vector<index_t> idx(pop.size()); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&fitness](auto i, auto j) {
        return fitness[i] > fitness[j];
    });
    std::for_each(fitness_updated.begin(), fitness_updated.end(), [m=*std::min_element(fitness_updated.begin(), fitness_updated.end())](auto& x) {
        x -= m;
    });
    if(float s = s=std::accumulate(fitness_updated.begin(), fitness_updated.end(), 0.0f); std::abs(s) < std::numeric_limits<float>::epsilon()) {
        std::for_each(fitness_updated.begin(), fitness_updated.end(), [s=1.0f/p.size()] (auto& x) {
            x = s;
        });
    } else {
        std::for_each(fitness_updated.begin(), fitness_updated.end(), [s] (auto& x) {
            x /= s;
        });
    }

    for(auto i = 0; i < pop.size(); i++) {
        if(i < elitism) {
            p[i] = pop[idx[i]];
            continue;
        }
        auto r = random_generator::random<float>();
        float range = 0;
        index_t j;
        for(j = 0; j < pop.size(); j++) {
            if(range <= r && r < range + fitness_updated[j]) {
                break;
            }
            range += fitness_updated[j];
        }
        if(specie[j].empty()) p[i] = pop[j];
        else {
            float range_ = 0;
            index_t k = specie[j][0];
            r = random_generator::random_uniform<float>(0, std::accumulate(specie[j].begin(), specie[j].end(), 0.0f, [&fitness_updated](auto a, auto b) {
                return a + fitness_updated[b];
            }));
            for(index_t l = 0; l < specie[j].size(); k = specie[j][++l]) {
                if(range_ <= r && r < range_ + fitness_updated[k]) {
                    break;
                }
                range_ += fitness_updated[k];
            }
            p[i] = genetic::crossover(pop[j], pop[k]);
        }
    }
    return p;
}



#endif //NEAT_NEAT_GENOMES_H
