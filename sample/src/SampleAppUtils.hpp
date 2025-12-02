#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"

namespace rtr::sample {

namespace fs = std::filesystem;

/// Resolves a path candidate relative to a list of preferred bases.
/// Mirrors the CLI sample search order so other entry points (e.g. the
/// on-screen viewer) can share the same logic when looking up config,
/// assets, or shader libraries.
std::optional<fs::path> resolvePath(const fs::path& candidate,
                                    bool requireDirectory,
                                    const std::vector<fs::path>& preferredBases);

/// Loads the engine config, falling back to sensible defaults when the
/// requested file is missing. Warnings are logged in the same style as the
/// CLI sample for consistency.
rtr::core::EngineConfig loadEngineConfig(const fs::path& configPath);

/// Builds a demo scene using the same helpers as the CLI sample. This keeps
/// the CLI and on-screen renderer in lockstep so both exercise the same
/// geometry/material setup.
rtr::scene::Scene buildScene(const std::string& sceneName, const fs::path& assetRoot);

}  // namespace rtr::sample
