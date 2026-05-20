#pragma once
#include <vector>
#include <array>
#include <utility>
#include "matrixMultiplication/MatrixMultiplication.h"
#include "common/vulkan/VulkanUtility.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    template<typename FloatType>
    class ImplVulkan {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;


        vulkan_utility::VulkanManager manager;
        std::vector<uint32_t> shader;
        std::shared_ptr<kp::Sequence> sequence;

        ImplVulkan();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
