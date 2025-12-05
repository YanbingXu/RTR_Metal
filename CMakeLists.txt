cmake_minimum_required(VERSION 3.21)

project(RTRMetal
    VERSION 0.1.0
    DESCRIPTION "Hardware ray tracing engine prototype built with C++ and Metal"
    LANGUAGES CXX OBJCXX)

include(FetchContent)

# Build options
option(RTR_BUILD_SAMPLE "Build RTR sample application" ON)
option(RTR_BUILD_ONSCREEN_SAMPLE "Build RTR on-screen sample (MetalKit UI)" ON)
option(RTR_BUILD_TESTS "Build RTR unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shader compilation -------------------------------------------------------
set(RTR_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(RTR_SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${RTR_SHADER_OUTPUT_DIR})
set(RTR_METAL_MODULE_CACHE ${RTR_SHADER_OUTPUT_DIR}/metal_module_cache)
file(MAKE_DIRECTORY ${RTR_METAL_MODULE_CACHE})

set(RTR_METAL_SOURCES
    ${RTR_SHADER_DIR}/RTRRayTracing.metal)

set(RTR_METALLIB ${RTR_SHADER_OUTPUT_DIR}/RTRShaders.metallib)

set(RTR_METAL_AIRS)
foreach(SOURCE_FILE ${RTR_METAL_SOURCES})
    get_filename_component(SOURCE_NAME ${SOURCE_FILE} NAME_WE)
    set(AIR_FILE ${RTR_SHADER_OUTPUT_DIR}/${SOURCE_NAME}.air)
    list(APPEND RTR_METAL_AIRS ${AIR_FILE})
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun --sdk macosx metal -std=metal3.0 -fmodules-cache-path=${RTR_METAL_MODULE_CACHE}
                -I${CMAKE_SOURCE_DIR}/engine/include -DMTL_ENABLE_RAYTRACING=1
                -c ${SOURCE_FILE} -o ${AIR_FILE}
        DEPENDS ${SOURCE_FILE}
        WORKING_DIRECTORY ${RTR_SHADER_DIR}
        COMMENT "Compiling ${SOURCE_NAME}.metal"
        VERBATIM)
endforeach()

add_custom_command(
    OUTPUT ${RTR_METALLIB}
    COMMAND xcrun --sdk macosx metallib ${RTR_METAL_AIRS} -o ${RTR_METALLIB}
    DEPENDS ${RTR_METAL_AIRS}
    COMMENT "Linking Metal shaders into ${RTR_METALLIB}"
    VERBATIM)

add_custom_target(RTRMetalShaders ALL DEPENDS ${RTR_METALLIB})

# Library ------------------------------------------------------------------
add_library(RTRMetalEngine STATIC
    engine/src/Core/ConfigLoader.cpp
    engine/src/Core/FileSystem.cpp
    engine/src/Core/Math.cpp
    engine/src/Core/Logger.cpp
    engine/src/Core/ImageLoader.mm
    engine/src/Rendering/AccelerationStructure.mm
    engine/src/Rendering/AccelerationStructureBuilder.mm
    engine/src/Rendering/RayTracingPipeline.mm
    engine/src/Rendering/BufferAllocator.mm
    engine/src/Rendering/GeometryStore.cpp
    engine/src/Rendering/MetalContext.mm
    engine/src/Rendering/Renderer.mm
    engine/src/MPS/MPSSceneConverter.cpp
    engine/src/Scene/Mesh.cpp
    engine/src/Scene/Scene.cpp
    engine/src/Scene/SceneBuilder.cpp
    engine/src/Scene/OBJLoader.cpp
    engine/src/Scene/DemoScenes.cpp
    engine/src/Scene/CornellBox.cpp)

target_include_directories(RTRMetalEngine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)

add_dependencies(RTRMetalEngine RTRMetalShaders)

target_compile_features(RTRMetalEngine PUBLIC cxx_std_20)

find_library(RTR_METAL_FRAMEWORK Metal REQUIRED)
find_library(RTR_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(RTR_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(RTR_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(RTR_IMAGEIO_FRAMEWORK ImageIO REQUIRED)

target_link_libraries(RTRMetalEngine
    PUBLIC
        ${RTR_METAL_FRAMEWORK}
        ${RTR_FOUNDATION_FRAMEWORK}
        ${RTR_MPS_FRAMEWORK}
        ${RTR_COREGRAPHICS_FRAMEWORK}
        ${RTR_IMAGEIO_FRAMEWORK})

set_source_files_properties(
    engine/src/Rendering/MetalContext.mm
    engine/src/Rendering/BufferAllocator.mm
    engine/src/Rendering/AccelerationStructure.mm
    engine/src/Rendering/AccelerationStructureBuilder.mm
    engine/src/Rendering/RayTracingPipeline.mm
    engine/src/Core/ImageLoader.mm
    engine/src/Rendering/Renderer.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Sample application -------------------------------------------------------
if(RTR_BUILD_SAMPLE)
    add_executable(RTRMetalSample
        sample/src/main.cpp
        sample/src/SampleAppUtils.cpp)
    target_link_libraries(RTRMetalSample PRIVATE RTRMetalEngine)
    target_include_directories(RTRMetalSample PRIVATE sample/src)
    add_dependencies(RTRMetalSample RTRMetalShaders)
    target_compile_features(RTRMetalSample PRIVATE cxx_std_20)
    target_compile_definitions(RTRMetalSample PRIVATE RTR_SOURCE_DIR="${CMAKE_SOURCE_DIR}" RTR_BINARY_DIR="${CMAKE_BINARY_DIR}")
endif()

if(RTR_BUILD_ONSCREEN_SAMPLE)
    add_executable(RTRMetalOnScreenSample MACOSX_BUNDLE
        sample/src/onscreen/main.mm
        sample/src/onscreen/AppDelegate.h
        sample/src/onscreen/AppDelegate.mm
        sample/src/onscreen/ViewController.h
        sample/src/onscreen/ViewController.mm
        sample/src/SampleAppUtils.cpp)
    target_link_libraries(RTRMetalOnScreenSample PRIVATE RTRMetalEngine "-framework Cocoa" "-framework MetalKit" "-framework QuartzCore")
    target_compile_features(RTRMetalOnScreenSample PRIVATE cxx_std_20)
    target_include_directories(RTRMetalOnScreenSample PRIVATE sample/src)
    target_compile_definitions(RTRMetalOnScreenSample PRIVATE RTR_SOURCE_DIR="${CMAKE_SOURCE_DIR}" RTR_BINARY_DIR="${CMAKE_BINARY_DIR}")
    add_dependencies(RTRMetalOnScreenSample RTRMetalShaders)
    add_custom_command(TARGET RTRMetalOnScreenSample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:RTRMetalOnScreenSample>/Resources
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${RTR_METALLIB}
                $<TARGET_BUNDLE_CONTENT_DIR:RTRMetalOnScreenSample>/Resources/RTRShaders.metallib
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/assets
                $<TARGET_BUNDLE_CONTENT_DIR:RTRMetalOnScreenSample>/Resources/assets)
endif()

# Tests --------------------------------------------------------------------
if(RTR_BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(RTRMetalTests
        tests/src/core/ConfigLoaderTests.cpp
        tests/src/core/EngineConfigTests.cpp
        tests/src/core/LoggerTests.cpp
        tests/src/core/MathTests.cpp
        tests/src/rendering/BufferAllocatorTests.cpp
        tests/src/rendering/GeometryStoreTests.cpp
        tests/src/rendering/AccelerationStructureBuilderTests.cpp
        tests/src/rendering/AccumulationTests.cpp
        tests/src/rendering/RendererTests.cpp
        tests/src/scene/SceneTests.cpp
        tests/src/scene/SceneBuilderTests.cpp
        tests/src/scene/ScenesTests.cpp
        tests/src/mps/MPSSceneConverterTests.cpp)
    target_link_libraries(RTRMetalTests
        PRIVATE RTRMetalEngine GTest::gtest_main)
    add_dependencies(RTRMetalTests RTRMetalShaders)
    include(GoogleTest)
    gtest_discover_tests(RTRMetalTests)
endif()

# Packaging / configuration ------------------------------------------------
install(TARGETS RTRMetalEngine)
if(RTR_BUILD_SAMPLE)
    install(TARGETS RTRMetalSample)
endif()
