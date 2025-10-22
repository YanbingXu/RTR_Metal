#include <iostream>

#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"

int main() {
    rtr::core::EngineConfig config{
        .applicationName = "RTR Metal Sample",
        .shaderLibraryPath = "shaders/RTRShaders.metallib",
    };

    rtr::rendering::Renderer renderer{config};
    renderer.renderFrame();

    std::cout << "Sample application executed successfully." << std::endl;
    return 0;
}
