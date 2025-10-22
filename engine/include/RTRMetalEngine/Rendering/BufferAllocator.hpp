#pragma once

#include <cstddef>
#include <memory>

namespace rtr::rendering {

class MetalContext;

class BufferHandle {
public:
    BufferHandle();
    ~BufferHandle();

    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    BufferHandle(BufferHandle&&) noexcept;
    BufferHandle& operator=(BufferHandle&&) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] std::size_t length() const noexcept;
    [[nodiscard]] void* nativeHandle() const noexcept;

private:
    class Impl;
    explicit BufferHandle(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class BufferAllocator;
};

class BufferAllocator {
public:
    explicit BufferAllocator(MetalContext& context);

    [[nodiscard]] bool isDeviceAvailable() const noexcept;
    [[nodiscard]] BufferHandle createBuffer(std::size_t length,
                                           const void* initialData = nullptr,
                                           const char* label = nullptr) noexcept;

private:
    MetalContext& context_;
};

}  // namespace rtr::rendering
