#include <gtest/gtest.h>

#include "RTRMetalEngine/Scene/CornellBox.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"

TEST(CornellBoxSceneTests, BuildsExpectedCounts) {
    rtr::scene::Scene scene = rtr::scene::createCornellBoxScene();
    EXPECT_GE(scene.meshes().size(), 7u);
    EXPECT_GE(scene.instances().size(), 7u);
    EXPECT_GE(scene.materials().size(), 5u);
}
