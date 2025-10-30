#pragma once

#include <filesystem>

#include "RTRMetalEngine/Scene/Scene.hpp"

namespace rtr::scene {

Scene createReflectiveDemoScene(const std::filesystem::path& assetRoot);
Scene createGlassDemoScene(const std::filesystem::path& assetRoot);

}  // namespace rtr::scene
