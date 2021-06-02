/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Types and enums that are shared between shaders and the host app code.
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

enum TextureIndices {
    kTextureY = 0,
    kTextureCbCr = 1,
    kTextureDepth = 2,
    kTextureConfidence = 3
};

enum BufferIndices {
    kPointCloudUniforms = 0,
    kParticleUniforms = 1,
    kGridPoints = 2,
    kCfdUniforms = 3,
    kCfdPoints = 4,
    kCfdIndices = 5,
    kCfdPitchInterval = 6,
    kMarkerAnchorVertex = 7,
    kColorbarVertex = 8,
    kAerofoilVertex = 9,
};

struct RGBUniforms {
    matrix_float3x3 viewToCamera;
    float viewRatio;
    float radius;
};

struct PointCloudUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 localToWorld;
    matrix_float3x3 cameraIntrinsicsInversed;
    simd_float2 cameraResolution;
    
    float particleSize;
    int maxPoints;
    int pointCloudCurrentIndex;
    int confidenceThreshold;
};

struct ParticleUniforms {
    simd_float3 position;
    simd_float3 color;
    float confidence;
};

struct MyVertex {
    simd_float3 position;
    simd_float3 color;
};

struct ColorbarVertex {
    simd_float2 position;
    float val;
};

enum MeshTypes {
    kPstat = 0,
    kPstag = 1,
    kVx = 2,
    kVt = 3
};

struct CfdUniforms {
    float pitch;
    float bladeStartX;
    float bladeStartY;
    float bladeEndX;
    float bladeEndY;
    float pstatMin;
    float pstatMax;
    float pstagMin;
    float pstagMax;
    float vxMin;
    float vxMax;
    float vtMin;
    float vtMax;
    int meshType;
    int showGridlines;
};

struct CfdPoint {
    int i;
    int j;
    float x;
    float y;
    float pstat;
    float pstag;
    float vx;
    float vt;
};

#endif /* ShaderTypes_h */
