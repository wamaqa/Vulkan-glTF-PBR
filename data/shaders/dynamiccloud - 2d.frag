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

struct camera
{
	vec3 pos;
	vec3 dir;
	vec3 up;
	vec3 right;
};



layout (set = 0, binding = 0) uniform sampler2D baseTexture;
layout (set = 0, binding = 1) uniform sampler2D noiseTexture1;
layout (set = 0, binding = 2) uniform sampler2D noiseTexture2;
layout (set = 0, binding = 3) uniform sampler2D noiseTexture3;
layout (set = 0, binding = 4) uniform sampler2D weatherTexture1;
layout (set = 0, binding = 5) uniform sampler2D weatherTexture2;

layout (push_constant) uniform CloudParams {
	float level;
	float height;
	float resolutionX;
    float resolutionY;
	float time;
	float speed;
} cloudParams;

layout (set = 1, binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 model;
	mat4 view;
	vec3 camPos;
} ubo;

layout (set = 1, binding = 1) uniform Params 
{
		vec4 lightDir;
		float exposure;
		float gamma;
		float prefilteredCubeMipLevels;
		float scaleIBLAmbient;
		float debugViewInputs;
		float debugViewEquation;
} lightParams;


layout (location = 0) in vec4 inFragPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec4 worldPos;
layout (location = 4) in camera inCamera;
layout (location = 9) in vec4 bottomPlane;
layout (location = 10) in vec4 topPlane;

layout (location = 0) out vec4 outColor;


vec4 _time = vec4(1.0);//时间片段
vec4 _weather_st = vec4(1.0);//时间片段


float uvScale = 0.025;
float beerPowder = 0.65;
float g = 0.65;
float lightAbsorption = 1.0;
float absorption = 1.0;

vec4 lightColor0 = vec4(1.0,1.0,1.0,1.0);

const float cloudscale = 1.1;
const float speed = 0.03;
const float clouddark = 0.5;
const float cloudlight = 0.3;
const float cloudcover = 0.2;
const float cloudalpha = 8.0;
const float skytint = 0.5;
const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);

const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

float fbm(vec2 n) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += noise(n) * amplitude;
		n = m * n;
		amplitude *= 0.4;
	}
	return total;
}
vec4 GetWorldPositionFromDepth(mat4 invVp, vec2 uv, float depth)
{
	vec4 wpos = invVp * vec4(uv, depth, 1.0);
//	wpos.xyz *= wpos.w;
//	wpos.y = -wpos.y;
	return wpos;
}

bool RayIntersectSphere(vec3 origin, vec3 dir, vec3 center, float radius,out vec3 target, out vec2 uv) 
{
		vec3 _vector = center - origin;
		float tca =dot(_vector, dir);// _vector.dot( this.direction );
		float d2 =dot(_vector, _vector) - tca * tca;
		float radius2 = radius * radius;
		if ( d2 > radius2 ) return false;

		float thc = sqrt( radius2 - d2 );

		// t0 = first intersect point - entrance on front of sphere
		float t0 = tca - thc;

		// t1 = second intersect point - exit point on back of sphere
		float t1 = tca + thc;

		// test to see if t1 is behind the ray - if so, return null
		if ( t1 < 0 ) return false;

		// test to see if t0 is behind the ray:
		// if it is, the ray is inside the sphere, so return the second exit point scaled by t1,
		// in order to always return an intersect point that is in front of the ray.
		if ( t0 < 0 ) 
		{
			target = origin + dir * t1;
		}
		else
		{
			target = origin + dir * t0;
		}
		if(target.y > 0) //盖帽
			return false;
		vec3 d = target - center;
		uv.x = d.x/ radius;
		uv.y = d.z/ radius;
		return true;
}

void main()
{
    vec2 resolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
	vec2 uv = gl_FragCoord.xy;
	uv.x = (uv.x / resolution.x) * 2 - 1;
	uv.y = (uv.y / resolution.y) * 2 - 1;
	uv.y = uv.y;
	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
	vec3 dirLength = (worldRayPos - ubo.camPos);
	vec3 rayDir = dirLength / length(dirLength);


	vec3 start = vec3(0,0,0);
	vec3 end = vec3(0,0,0);
	vec2 mapUv;
	bool isInsertStart = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 500.0,start,mapUv);
	bool isInsertEnd = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 600.0,end,mapUv);

	uv = mapUv;

//	vec2 p = gl_FragCoord.xy / resolution;
    float time = cloudParams.time * cloudParams.speed;
    float q = fbm(uv * cloudscale * 0.5);
    
	float r = 0.0;
	uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.7;
    }
    
	float f = 0.0;
    uv = mapUv;
	//uv = p * vec2(resolution.x/resolution.y,1.0);
	uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    float c = 0.0;
    time =cloudParams.time * cloudParams.speed * 2.0;
    uv = mapUv;
	// uv = p*vec2(resolution.x/resolution.y,1.0);
	uv *= cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    float c1 = 0.0;
    time = cloudParams.time * cloudParams.speed * 3.0;
    uv = mapUv;//uv = p*vec2(resolution.x/resolution.y,1.0);
	uv *= cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c += c1;
    
    vec3 skycolour = mix(skycolour2, skycolour1, mapUv.y);//p.y);
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
   
    f = cloudcover + cloudalpha*f*r;
    
    vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    
	outColor = vec4( result, 1.0 );

//	vec2 boxCast = rayBoxDst(boundMin.xyz, boundMax.xyz, ubo.camPos, 1 / rayDir);
//
//	vec3 start = ubo.camPos + rayDir * boxCast.x;
//	vec3 end = ubo.camPos + rayDir * (boxCast.x + boxCast.y);
//	float v =  StepBox(boundMin.xyz,boundMax.xyz, start, end);
//
//	vec4 cloud = renderCloud(start, end);
//	outColor = cloud;//vec4(vec3(v), 1.0);
//v = NOISE(X * 10.0) * clamp(X.y * 0.75 + 1.0 - min(X.z * 0.05, 0.0), 0.0, 1.0) + 
//				clamp((length(X.xz) - 0.75) * 0.15, 0.0, 0.1);
//	vec3 start = vec3(0,0,0);
//	vec3 end = vec3(0,0,0);
//	vec2 mapUv;
//	bool isInsertStart = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 500.0,start,mapUv);
//	bool isInsertEnd = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 600.0,end,mapUv);
////	if(isInsert)
////	{
////		
////	}
//////	
////	bool isInsertStart = RayPlane(ubo.camPos, rayDir, bottomPlane,start);
//	if(isInsertStart)
//	{
//		vec4 cloud = renderCloud(start, end, mapUv);
//		float z = fbm(start);
//		vec3 col =texture(weatherTexture1, vec2(start.x, start.y)).xyz;
//		outColor = vec4(z); //texture(weatherTexture1, mapUv);//vec4(z);//vec4(col, length(col));
//	}
//	else
//	{
//		outColor =  vec4(0.0);
//	}
//	outColor = vec4(rayDir, 1.0);//vec4(z);
//	outColor =  vec4(fbm(uv));
//	outColor = vec4(rayDir, 1.0);
//	bool isInsertEnd = RayPlane(ubo.camPos, rayDir, topPlane,end);
//	vec4 base = texture(baseTexture, inUV.xy);
//	if(isInsertStart && isInsertEnd)
//	{
//		float val = StepRender(start,end);
//		outColor = vec4(0.8);
//	}
//	else
//	{
//		outColor =  vec4(0.0);//renderCloud(start,end);
//	}
	
}
