#include "RTRMetalEngine/Scene/OBJLoader.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

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

struct OBJVertexIndex {
    int position = -1;
    int texcoord = -1;
    int normal = -1;
};

int resolveIndex(int rawIndex, std::size_t count) {
    if (rawIndex > 0) {
        return rawIndex - 1;
    }
    if (rawIndex < 0) {
        return static_cast<int>(count) + rawIndex;
    }
    return -1;
}

bool parseFaceVertex(const std::string_view token,
                     std::size_t positionCount,
                     std::size_t texcoordCount,
                     std::size_t normalCount,
                     OBJVertexIndex& outIndex) {
    if (token.empty()) {
        return false;
    }

    std::array<int, 3> components = {-1, -1, -1};
    std::size_t componentIndex = 0;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= token.size(); ++i) {
        if (i == token.size() || token[i] == '/') {
            if (componentIndex >= components.size()) {
                break;
            }
            const std::size_t length = i - start;
            if (length > 0) {
                const std::string_view component = token.substr(start, length);
                try {
                    components[componentIndex] = std::stoi(std::string(component));
                } catch (const std::exception&) {
                    components[componentIndex] = -1;
                }
            }
            ++componentIndex;
            start = i + 1;
        }
    }

    outIndex.position = resolveIndex(components[0], positionCount);
    outIndex.texcoord = resolveIndex(components[1], texcoordCount);
    outIndex.normal = resolveIndex(components[2], normalCount);

    if (outIndex.position < 0 || static_cast<std::size_t>(outIndex.position) >= positionCount) {
        return false;
    }
    if (outIndex.texcoord >= static_cast<int>(texcoordCount)) {
        outIndex.texcoord = -1;
    }
    if (outIndex.normal >= static_cast<int>(normalCount)) {
        outIndex.normal = -1;
    }
    return true;
}

struct VertexKey {
    int position = -1;
    int texcoord = -1;
    int normal = -1;

    bool operator==(const VertexKey& other) const noexcept {
        return position == other.position && texcoord == other.texcoord && normal == other.normal;
    }
};

struct VertexKeyHasher {
    std::size_t operator()(const VertexKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.position + 1) * 73856093;
        hash ^= static_cast<std::size_t>(key.texcoord + 1) * 19349663;
        hash ^= static_cast<std::size_t>(key.normal + 1) * 83492791;
        return hash;
    }
};

}  // namespace

bool loadOBJMesh(const std::filesystem::path& path,
                 std::vector<Vertex>& outVertices,
                 std::vector<std::uint32_t>& outIndices) {
    outVertices.clear();
    outIndices.clear();

    std::ifstream file(path);
    if (!file.is_open()) {
        rtr::core::Logger::error("OBJLoader", "Failed to open %s", path.string().c_str());
        return false;
    }

    std::vector<simd_float3> positions;
    std::vector<simd_float3> normals;
    std::vector<simd_float2> texcoords;

    std::unordered_map<VertexKey, std::uint32_t, VertexKeyHasher> vertexRemap;

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
        } else if (tag == "vn") {
            float x = 0.0F;
            float y = 0.0F;
            float z = 0.0F;
            stream >> x >> y >> z;
            normals.push_back(simd_make_float3(x, y, z));
        } else if (tag == "vt") {
            float u = 0.0F;
            float v = 0.0F;
            stream >> u >> v;
            texcoords.push_back(simd_make_float2(u, v));
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

            std::array<OBJVertexIndex, 4> faceVertices{};
            for (std::size_t i = 0; i < tokenCount; ++i) {
                if (!parseFaceVertex(tokens[i], positions.size(), texcoords.size(), normals.size(), faceVertices[i])) {
                    rtr::core::Logger::warn("OBJLoader", "Failed to parse face vertex '%s'", tokens[i].c_str());
                    tokenCount = 0;
                    break;
                }
            }

            if (tokenCount < 3) {
                continue;
            }

            auto fetchIndex = [&](const OBJVertexIndex& idx) -> std::uint32_t {
                VertexKey key{idx.position, idx.texcoord, idx.normal};
                auto found = vertexRemap.find(key);
                if (found != vertexRemap.end()) {
                    return found->second;
                }

                Vertex vertex{};
                vertex.position = positions[static_cast<std::size_t>(idx.position)];
                if (idx.normal >= 0 && static_cast<std::size_t>(idx.normal) < normals.size()) {
                    vertex.normal = normals[static_cast<std::size_t>(idx.normal)];
                }
                if (idx.texcoord >= 0 && static_cast<std::size_t>(idx.texcoord) < texcoords.size()) {
                    vertex.texcoord = texcoords[static_cast<std::size_t>(idx.texcoord)];
                }

                const std::uint32_t newIndex = static_cast<std::uint32_t>(outVertices.size());
                outVertices.push_back(vertex);
                vertexRemap.emplace(key, newIndex);
                return newIndex;
            };

            // Triangulate fan with remapped vertices
            for (std::size_t tri = 1; tri + 1 < tokenCount; ++tri) {
                const std::uint32_t i0 = fetchIndex(faceVertices[0]);
                const std::uint32_t i1 = fetchIndex(faceVertices[tri]);
                const std::uint32_t i2 = fetchIndex(faceVertices[tri + 1]);
                outIndices.push_back(i0);
                outIndices.push_back(i1);
                outIndices.push_back(i2);
            }
        }
    }

    if (outVertices.empty() || outIndices.empty()) {
        rtr::core::Logger::warn("OBJLoader", "OBJ %s produced no triangles", path.string().c_str());
        return false;
    }

    auto isZeroNormal = [](const simd_float3& normal) {
        return std::abs(normal.x) < 1e-5F && std::abs(normal.y) < 1e-5F && std::abs(normal.z) < 1e-5F;
    };

    bool missingNormals = false;
    for (const auto& vertex : outVertices) {
        if (isZeroNormal(vertex.normal)) {
            missingNormals = true;
            break;
        }
    }

    if (missingNormals) {
        std::vector<simd_float3> accum(outVertices.size(), simd_make_float3(0.0F, 0.0F, 0.0F));
        std::vector<int> counts(outVertices.size(), 0);
        for (std::size_t i = 0; i + 2 < outIndices.size(); i += 3) {
            const std::uint32_t i0 = outIndices[i + 0];
            const std::uint32_t i1 = outIndices[i + 1];
            const std::uint32_t i2 = outIndices[i + 2];
            const simd_float3 p0 = outVertices[i0].position;
            const simd_float3 p1 = outVertices[i1].position;
            const simd_float3 p2 = outVertices[i2].position;
            const simd_float3 normal = simd_normalize(simd_cross(p1 - p0, p2 - p0));
            if (std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z)) {
                accum[i0] += normal;
                accum[i1] += normal;
                accum[i2] += normal;
                counts[i0]++;
                counts[i1]++;
                counts[i2]++;
            }
        }
        for (std::size_t i = 0; i < outVertices.size(); ++i) {
            if (counts[i] > 0) {
                const simd_float3 averaged = simd_normalize(accum[i]);
                if (std::isfinite(averaged.x) && std::isfinite(averaged.y) && std::isfinite(averaged.z)) {
                    outVertices[i].normal = averaged;
                }
            }
        }
    }

    return true;
}

}  // namespace rtr::scene
