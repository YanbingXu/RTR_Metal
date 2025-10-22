#pragma once

#include <memory>

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"

namespace rtr::scene {
class Scene;
}

namespace rtr::rendering {

class MetalContext;
class MPSRenderer {
public:
    explicit MPSRenderer(MetalContext& context);

    bool initialize();
    bool initialize(const scene::Scene& scene);
    bool renderFrame(const char* outputPath);

private:
    MetalContext& context_;
    MPSPathTracer pathTracer_;
};

}  // namespace rtr::rendering
