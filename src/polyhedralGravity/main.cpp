#include <benchmark/benchmark.h>
#include "PolyhedralGravityDefinitions.h"

static void BM_Eros(benchmark::State &state) {
    std::vector<Array3> Vertices;
    std::vector<IndexArray3> Faces;

    read_tetgen("data/Eros", Vertices, Faces);

    auto G = create_gravity_evaluable(Vertices, Faces, 2.0);

    for (auto _: state) {
        benchmark::DoNotOptimize(G->evaluate({1.0, 0, 0}));
    }

    state.counters["NumFaces"] = Faces.size();
}

BENCHMARK(BM_Eros);

template<class... Args>
void BM_obj(benchmark::State &state, Args &&...args) {
    std::vector<Array3> Vertices;
    std::vector<IndexArray3> Faces;

    auto args_tuple = std::make_tuple(std::move(args)...);
    read_obj("data/" + std::get<0>(args_tuple) + ".obj", Vertices, Faces);

    auto G = create_gravity_evaluable(Vertices, Faces, 2.0);

    for (auto _: state) {
        benchmark::DoNotOptimize(G->evaluate({1.0, 0, 0}));
    }

    state.counters["NumFaces"] = Faces.size();
}

BENCHMARK_CAPTURE(BM_obj, 67P_ESA_NAVCAM_Jul2015data_256k, std::string("67P_ESA_NAVCAM_Jul2015data_256k"));
BENCHMARK_CAPTURE(BM_obj, 25143_Itokawa_200k, std::string("25143_Itokawa_200k"));
BENCHMARK_CAPTURE(BM_obj, a8567, std::string("a8567.tab"));
// BENCHMARK_CAPTURE(BM_obj, SHAPE_SFM_3M_v20180804, std::string("SHAPE_SFM_3M_v20180804"));
// BENCHMARK_CAPTURE(BM_obj, 4179toutatis, std::string("4179toutatis.tab"));
// BENCHMARK_CAPTURE(BM_obj, hartley2_2012_cart, std::string("hartley2_2012_cart"));

// Based on https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/

using Lookup = std::map<std::pair<size_t, size_t>, size_t>;

size_t vertex_for_edge(Lookup &lookup, std::vector<Array3> &Vertices, size_t first, size_t second) {
    Lookup::key_type key(first, second);
    if (key.first > key.second)
        std::swap(key.first, key.second);

    auto inserted = lookup.insert({key, Vertices.size()});
    if (inserted.second) {
        auto &edge0 = Vertices[first];
        auto &edge1 = Vertices[second];
        auto point = (edge0 + edge1);
        point = point / euclideanNorm(point);

        Vertices.push_back(point);
    }

    return inserted.first->second;
}

static void subdivide(std::vector<Array3> &Vertices, std::vector<IndexArray3> &Faces) {
    Lookup lookup;
    std::vector<IndexArray3> NewFaces{};

    for (auto &Face: Faces) {
        IndexArray3 mid;
        for (int edge = 0; edge < 3; ++edge) {
            mid[edge] = vertex_for_edge(lookup, Vertices, Face[edge], Face[(edge + 1) % 3]);
        }
        NewFaces.push_back({Face[0], mid[0], mid[2]});
        NewFaces.push_back({Face[1], mid[1], mid[0]});
        NewFaces.push_back({Face[2], mid[2], mid[1]});
        NewFaces.push_back({mid[0], mid[1], mid[2]});
    }
    std::swap(Faces, NewFaces);
}

static void generate_sphere(std::vector<Array3> &Vertices, std::vector<IndexArray3> &Faces, int subdivisions) {
    const float X = .525731112119133606f;
    const float Z = .850650808352039932f;
    const float N = 0.f;

    Vertices = {
            {-X, N, Z},
            {X, N, Z},
            {-X, N, -Z},
            {X, N, -Z},
            {N, Z, X},
            {N, Z, -X},
            {N, -Z, X},
            {N, -Z, -X},
            {Z, X, N},
            {-Z, X, N},
            {Z, -X, N},
            {-Z, -X, N}};
    Faces = {
            {0, 4, 1},
            {0, 9, 4},
            {9, 5, 4},
            {4, 5, 8},
            {4, 8, 1},
            {8, 10, 1},
            {8, 3, 10},
            {5, 3, 8},
            {5, 2, 3},
            {2, 7, 3},
            {7, 10, 3},
            {7, 6, 10},
            {7, 11, 6},
            {11, 0, 6},
            {0, 1, 6},
            {6, 1, 10},
            {9, 0, 11},
            {9, 11, 2},
            {9, 2, 5},
            {7, 2, 11}};

    for (int i = 0; i < subdivisions; ++i) {
        subdivide(Vertices, Faces);
    }
}

int main(int argc, char **argv) {
    GlobalResources Resources(argc, argv);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
