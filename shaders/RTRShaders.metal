#include <metal_stdlib>
using namespace metal;

kernel void rayGenMain(device float4* output [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (!output) {
        return;
    }
    output[gid.x] = float4(0.0, 0.0, 0.0, 1.0);
}
