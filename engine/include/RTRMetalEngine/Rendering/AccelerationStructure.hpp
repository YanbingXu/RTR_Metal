#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace rtr::rendering {

class AccelerationStructure {
public:
    AccelerationStructure();
    AccelerationStructure(std::string label, std::size_t size, std::unique_ptr<class AccelerationStructureImpl> impl);
    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&&) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] const std::string& label() const noexcept { return label_; }
    [[nodiscard]] std::size_t sizeInBytes() const noexcept { return sizeInBytes_; }

    void* rawHandle() const noexcept;

private:
    std::string label_;
    std::size_t sizeInBytes_ = 0;
    std::unique_ptr<class AccelerationStructureImpl> impl_;
};

AccelerationStructure makeAccelerationStructure(std::string label, std::size_t size, void* handle);

}  // namespace rtr::rendering
