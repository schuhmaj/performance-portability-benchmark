#include <chrono>
#include <utility>
#include "Impl_SlangVulkan.h"
#include "common/UtilityFloatArithmetic.h"
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    ImplVulkan<FloatType>::ImplVulkan() : manager{}, shader{MATRIXMULTIPLICATIONSHADER_COMP_SPV.begin(), MATRIXMULTIPLICATIONSHADER_COMP_SPV.end()}, sequence{manager.sequence()}{}

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double>
    ImplVulkan<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,
                                     const MatrixMultiplicationConfig &config) {
        const size_t resultSize = config.m * config.n;
        std::vector<FloatType> result(resultSize, 0.0);

        auto tensorA = manager.tensor(a);
        auto tensorB = manager.tensor(b);
        auto tensorC = manager.tensor(result);

        std::vector<std::shared_ptr<kp::Tensor>> params = {tensorA, tensorB, tensorC};

        std::vector<unsigned int> pushConstants({static_cast<unsigned int>(config.m),static_cast<unsigned int>(config.n), static_cast<unsigned int>(config.k)});

        constexpr unsigned int TILE_SIZE = 32;
        const unsigned int groups_x = util::ceilDiv<unsigned int>(config.m, TILE_SIZE);
        const unsigned int groups_y = util::ceilDiv<unsigned int>(config.n, TILE_SIZE);
        kp::Workgroup workgroup{{groups_x, groups_y, 1}};

        auto algorithm = manager.algorithm(params, shader, workgroup, {}, pushConstants);

        sequence->template record<kp::OpTensorSyncDevice>(params)->eval();

        const auto start = std::chrono::high_resolution_clock::now();

        sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushConstants)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
        result = tensorC->vector();
        return std::make_pair(result, elapsed_nanoseconds);
    }

    template class ImplVulkan<float>;
} // namespace ppb