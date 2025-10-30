#include "RTRMetalEngine/Scene/OBJLoader.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

#include <array>
#include <cctype>
#include <fstream>
#include <sstream>

namespace rtr::scene {

namespace {

std::string_view trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

bool parseFaceVertex(const std::string_view token, std::size_t vertexCount, std::uint32_t& outIndex) {
    if (token.empty()) {
        return false;
    }

    std::string_view indexView = token;
    const auto slashPos = token.find('/');
    if (slashPos != std::string_view::npos) {
        indexView = token.substr(0, slashPos);
    }

    if (indexView.empty()) {
        return false;
    }

    int parsed = 0;
    try {
        parsed = std::stoi(std::string(indexView));
    } catch (const std::exception&) {
        return false;
    }

    int resolved = parsed;
    if (parsed < 0) {
        resolved = static_cast<int>(vertexCount) + parsed + 1;
    }

    if (resolved <= 0 || static_cast<std::size_t>(resolved) > vertexCount) {
        return false;
    }

    outIndex = static_cast<std::uint32_t>(resolved - 1);
    return true;
}

}  // namespace

bool loadOBJMesh(const std::filesystem::path& path,
                 std::vector<simd_float3>& outPositions,
                 std::vector<std::uint32_t>& outIndices) {
    outPositions.clear();
    outIndices.clear();

    std::ifstream file(path);
    if (!file.is_open()) {
        rtr::core::Logger::error("OBJLoader", "Failed to open %s", path.string().c_str());
        return false;
    }

    std::vector<simd_float3> positions;

    std::string line;
    while (std::getline(file, line)) {
        std::string_view view = trim(line);
        if (view.empty() || view.front() == '#') {
            continue;
        }

        std::istringstream stream{std::string(view)};
        std::string tag;
        stream >> tag;

        if (tag == "v") {
            float x = 0.0F;
            float y = 0.0F;
            float z = 0.0F;
            stream >> x >> y >> z;
            positions.push_back(simd_make_float3(x, y, z));
        } else if (tag == "f") {
            std::array<std::string, 4> tokens{};
            std::size_t tokenCount = 0;
            while (tokenCount < tokens.size() && (stream >> tokens[tokenCount])) {
                ++tokenCount;
            }

            if (tokenCount < 3) {
                rtr::core::Logger::warn("OBJLoader", "Skipping face with <3 vertices in %s", path.string().c_str());
                continue;
            }

            std::array<std::uint32_t, 4> indices{};
            for (std::size_t i = 0; i < tokenCount; ++i) {
                if (!parseFaceVertex(tokens[i], positions.size(), indices[i])) {
                    rtr::core::Logger::warn("OBJLoader", "Failed to parse face vertex '%s'", tokens[i].c_str());
                    tokenCount = 0;
                    break;
                }
            }

            if (tokenCount < 3) {
                continue;
            }

            // Triangulate fan
            for (std::size_t tri = 1; tri + 1 < tokenCount; ++tri) {
                outIndices.push_back(indices[0]);
                outIndices.push_back(indices[tri]);
                outIndices.push_back(indices[tri + 1]);
            }
        }
    }

    outPositions = std::move(positions);

    if (outPositions.empty() || outIndices.empty()) {
        rtr::core::Logger::warn("OBJLoader", "OBJ %s produced no triangles", path.string().c_str());
        return false;
    }

    return true;
}

}  // namespace rtr::scene
