#pragma once

#include "RTRMetalEngine/Scene/Scene.hpp"

#include <filesystem>

namespace rtr::scene {

Scene createCornellBoxScene();
Scene createCornellBoxScene(const std::filesystem::path& assetRoot);

}  // namespace rtr::scene
