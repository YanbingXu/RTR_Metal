#pragma once

#include <memory>
#include <string>

namespace rtr::rendering {

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) noexcept;
    MetalContext& operator=(MetalContext&&) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] const std::string& deviceName() const noexcept;
    [[nodiscard]] bool supportsRayTracing() const noexcept;

    void logDeviceInfo() const;

    [[nodiscard]] void* rawDeviceHandle() const noexcept;
    [[nodiscard]] void* rawCommandQueue() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    friend class BufferAllocator;
    friend class AccelerationStructureBuilder;
};

}  // namespace rtr::rendering
