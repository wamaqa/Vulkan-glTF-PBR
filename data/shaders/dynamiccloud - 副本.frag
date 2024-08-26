/* Copyright (c) 2018-2024, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

// PBR shader based on the Khronos WebGL PBR implementation
// See https://github.com/KhronosGroup/glTF-WebGL-PBR
// Supports both metallic roughness and specular glossiness inputs

#version 450

//#extension GL_KHR_vulkan_glsl : require
//#extension GL_GOOGLE_include_directive : require

layout (set = 0, binding = 0) uniform sampler2D baseTexture;
layout (set = 0, binding = 1) uniform sampler2D noiseTexture1;
layout (set = 0, binding = 2) uniform sampler2D noiseTexture2;
layout (set = 0, binding = 3) uniform sampler2D noiseTexture3;
layout (set = 0, binding = 4) uniform sampler2D weatherTexture1;
layout (set = 0, binding = 4) uniform sampler2D weatherTexture2;

layout (location = 0) out vec4 outColor;


layout (push_constant) uniform CloudParams {
	float height;
	float resolutionX;
    float resolutionY;
} cloudParams;

layout (set = 0, binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 model;
	mat4 view;
	vec3 camPos;
} ubo;





//uniform vec2 u_resolution;
//uniform vec2 u_mouse;
//uniform float u_time;

float random (in vec2 st) {
    float key = dot(st.xy, vec2(12.9898,78.233));
    float kv = sin(key) * 10.5453123; //不能搞太大 会溢出
    return fract(kv);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float perlinNoise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 0.0;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * perlinNoise(st);
        st *= 3.0;
        amplitude *= .5;
    }
    return value;
}

vec3 getRayDir(vec3 ro, vec3 point)
{
	return normalize(ro-point);
}



float rayCloud(vec3 ray, vec3 dir, float height)
{
    return dir.z;
}

void main()
{
    vec2 resolution = vec2(cloudParams.resolutionX,cloudParams.resolutionY);
    vec2 st = gl_FragCoord.xy/resolution;
    st.x *= resolution.x/resolution.y;
    vec4 baseColor = fbm(st * 3.0) * vec4(1.0,1.0,1.0,1.0);

    if(baseColor.z > 1.0) baseColor.z = 1.0;
    outColor = baseColor;


////	vec3 rayOrigin = vec3(0.0,2.0,0);
////    float x = gl_FragCoord.x - cloudParams.resolution.x/2.0;
////    float y = gl_FragCoord.y - cloudParams.resolution.y/2.0;
////    vec3 sPoint = vec3(x, 0.0, -y);
////
//////    vec2 sc = gl_FragCoord.xy/cloudParams.resolution;
////
////	vec3 rayDir = getRayDir(rayOrigin, sPoint);
////    
////
////    if(rayCloud(rayOrigin, rayDir, cloudParams.height) > 0)
////    {
////        float le = length(gl_FragCoord.xy/resolution);
//	    vec3 col = random(st) * vec3(1.0);
//	    outColor = vec4(col,1.0);
////    }
	
}
