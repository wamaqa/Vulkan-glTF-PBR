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

vec2 rayBoxDst(vec3 boundsMin, vec3 boundsMax, vec3 rayOrigin, vec3 invRaydir)
{
	vec3 t0 = (boundsMin - rayOrigin) * invRaydir;
	vec3 t1 = (boundsMax - rayOrigin) * invRaydir;
	vec3 tmin = min(t0, t1);
	vec3 tmax = max(t0, t1);

	float dstA = max(max(tmin.x, tmin.y), tmin.z); //进入点
	float dstB = min(tmax.x, min(tmax.y, tmax.z)); //出去点

	float dstToBox = max(0, dstA);
	float dstInsideBox = max(0, dstB - dstToBox);
	return vec2(dstToBox, dstInsideBox);
}

vec4 GetWorldPositionFromDepth(mat4 invVp, vec2 uv, float depth)
{
	vec4 wpos = invVp * vec4(uv, depth, 1.0);
//	wpos.xyz *= wpos.w;
//	wpos.y = -wpos.y;
	return wpos;
}

float remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
	return new_min + (((original_value - original_min) / (original_max - original_min)) * (new_max - new_min));
}

//float random(vec2 st) 
//{ 
//    return frac(sin(dot(st.xy, vec2(12.9898, 78.233)))* 43758.5453123);
//}
//
//uniform vec2 u_resolution;
//uniform vec2 u_mouse;
//uniform float u_time;

float random (in vec2 st) {
    float key = dot(st.xy, vec2(12.9898,78.233));
    float kv = sin(key) * 10.5453123; //不能搞太大 会溢出
    return fract(kv);
}

vec2 computeCurl(vec2 st)
{
	float x = st.x; float y = st.y;
	float h = 0.0001;
	float n, n1, n2, a, b;

	n = random(vec2(x, y));
	n1 = random(vec2(x, y - h));
	n2 = random(vec2(x - h, y));
	a = (n - n1) / h;
	b = (n - n2) / h;

	return vec2(a, -b);
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

float saturate(float val)
{
	if(val < 0.0 ) return 0.0;
	if(val > 0.0 ) return 1.0;
	return val;
}

float GetDensityHeightGradientForPoint1(float h, float w)
{
	float d = remap(h, 0, w, 1, 0);
	return saturate(d)*2;
}


float cloudShape(vec3 pos,vec2 uv,float heightPercent)
{
	uv.x += _time.y*0.004;
	vec4 wColor = texture(weatherTexture2, uv);
	float weather = length(wColor);//vec4( uv*_weather_st.xy+_weather_st.zw,0,0));
	float heightFra = GetDensityHeightGradientForPoint1(heightPercent, weather);

	vec3 detailUVW = pos * uvScale;
	detailUVW.xy += vec2( _time.y*0.04,_time.w*0.01);
	vec4 low_frequency_noises = texture(noiseTexture1, detailUVW.xy);
	// 从低频Worley噪声中构建FBM，可用于为低频Perlin-Worley噪声添加细节。 
	float low_freq_FBM = (low_frequency_noises.g * 0.625) + (low_frequency_noises.b*0.25) + (low_frequency_noises.r * 0.125);
	// 通过使用由Worley噪声制成的低频FBM对其进行膨胀来定义基本云形状。
	float base_cloud = remap(low_frequency_noises.r, -(1.0 - low_freq_FBM), 1.0, 0.0, 1.0);
	base_cloud *= heightFra;
	//return base_cloud;

	float cloud_coverage = weather.x*0.9;
	// 使用重新映射来应用云覆盖属性。 
	float base_cloud_with_coverage = saturate(remap(base_cloud, cloud_coverage, 1.0, 0.0, 1.0));
	// 将结果乘以云覆盖属性，使较小的云更轻且更美观。 
	base_cloud_with_coverage *= cloud_coverage;

	//base_cloud_with_coverage = base_cloud;

	vec2 curl_noise = normalize(computeCurl(uv));
	//在云底添加一些湍流。
	pos.xy += curl_noise.xy *(1.0 - heightPercent);
	//采样高频噪声。 
	vec3 high_frequency_noises = texture(noiseTexture1, detailUVW.xy).rgb;
	//构建高频Worley噪声FBM。 
	float high_freq_FBM = (high_frequency_noises.r * 0.625) + (high_frequency_noises.g *0.25) + (high_frequency_noises.b * 0.125);
	//return high_freq_FBM;

	//从纤细的形状过渡到高度的波浪形状。
	float high_freq_noise_modifier = mix(high_freq_FBM, 1.0 - high_freq_FBM, saturate(heightPercent * 10.0));	//用扭曲的高频Worley噪声侵蚀基础云形状。 
	float final_cloud = saturate(remap(base_cloud_with_coverage, high_freq_noise_modifier * 0.2, 1.0, 0.0, 1.0));
	return final_cloud;
}

float sampleCloud(vec3 pos,vec2 uv,float heightPercent)
{
	return cloudShape(pos, uv, heightPercent);
}


float HenyeyGreenstein(float cosine)
{
	float coeff =1.0;// _G;
	float g2 = coeff * coeff;
	return (1 - g2) / (4 * 3.1415*pow(1 + g2 - 2 * coeff * cosine, 1.5));
}

float Beer(float depth)
{
	return exp(depth);
}

float BeerPowder(float depth)
{
	float e = 1.0;//_BP;
	return exp(-e * depth) * (1 - exp(-e * 2 * depth))*2;
}

float SampleCloudDensityAlongCone(vec3 p,vec2 uv,float heightPercent)
{
	vec3 lightDir = -normalize(worldPos.xyz);
	float dis = 100;//rayBoxDst(boundMin.xyz, boundMax.xyz, p, 1 / lightDir).y;
	float stepSize = dis / 6;
	float cone_spread_multiplier = stepSize;
	vec3 light_step = normalize(lightDir)* stepSize;
	float density_along_cone = 0.0;
	//光照的ray-march循环。 
	for (float i = 0; i <= 6; i++)
	{
		vec3 noise = texture(noiseTexture3, p.xz).rgb;
		//将采样位置加上当前步距。 
		p += light_step + (cone_spread_multiplier * noise * i);
		density_along_cone += sampleCloud(p, uv, heightPercent)*stepSize;
	}
	return Beer(density_along_cone);
}

float sampleLight(vec3 p,vec2 uv,float heightPercent)
{
	vec3 lightDir = normalize(worldPos.xyz);
	//float dis = rayBoxDst(boundMin.xyz, boundMax.xyz, p, 1 / lightDir).y;
	float stepSize = 2;//dis / 8;
	//p = p - lightDir * dis;
	vec3 light_step = lightDir * stepSize;
	float density = 0;
	//光照的ray-march循环。 
	for (float i = 0; i <= 8; i++)
	{
		//将采样位置加上当前步距。 
		p += light_step;
		density += sampleCloud(p, uv, heightPercent)*stepSize;
	}
	return Beer(density* -lightAbsorption);

}

vec4 renderCloud(vec3 start,vec3 end, vec2 uv)
{
	float sum = 1.0;
	float stepCount = 128;

	vec3 dir = normalize(end - start);
	float dis = length(end - start);
	float stepSize = dis / stepCount;

	vec3 samplePos = start;
	float d = dot(normalize(worldPos.xyz), normalize(dir));
	float hg = HenyeyGreenstein(d);
	float light = 0;
	float heightPercent = 0;
	for (float i = 0; i < stepCount; i++)
	{
		float density = sampleCloud(samplePos, uv, i/ stepCount);
		if (density > 0.0)
		{
			light += sampleLight(samplePos,uv, i/ stepCount)* density*BeerPowder(sum)*stepSize;//*(exp(-i/stepCount) / 4.75)
			sum *= Beer(density*stepSize* -absorption);
		}
		if (sum <= 0.01)
			break;
		//samplePos = start + dir * stepSize * (i + random(samplePos.xz));
		samplePos += dir * stepSize;
	}
	light =saturate(light * (hg + 0.45));
	return vec4(mix(vec3(0.2, 0.2, 0.2), lightColor0.xyz, light), 1 - sum);
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
		if(target.y > 0)
			return false;
		//投影式uv；
		// calculate u,v coordinates of intersection point
//		vec3 d = normalize(target - center);
//		d = d / length(d);
//		uv.x = atan(d.y/d.x)/2/3.141592654;
//		uv.y = asin(d.z) / 3.141592654;
//		V=arcsin(z/r)/PI+0.5，
//		U=arctan(y/x)/2/PI
//
//		vec2 xz = normalize(d.xz);
//		uv.x = acos(xz.x);
//		if(xz.y < 0)
//		{	
//			uv.x = 6.28318530718 - uv.x;
//		}
//
//	
//		uv.y = asin(length(d.xz));
//		if(d.z < 0)
//		{	
//			uv.y = 6.28318530718 - uv.y;
//		}
//
//
//		uv.x = uv.x / 6.28318530718;
//	    uv.y = uv.y / 6.28318530718;
//		// else t0 is in front of the ray, so return the first collision point scaled by t0
		return true;
}

bool RayPlane(vec3 origin, vec3 dir, vec4 plane, out vec3 insert)
{
		float denominator =dot(plane.xyz, dir);

		if ( denominator == 0 ) {
			
			// line is coplanar, return origin
			
			if (dot(plane.xyz, origin ) + plane.w== 0 ) {
				insert = origin;
				return true;
			}
			return false;

		}

		float t = - ( dot(origin, plane.xyz) + plane.w ) / denominator;
		if(t<0)
		{
			return false;
		}
		insert = origin + dir * t;

		return true;
}
float RenderShape(vec3 pos)
{
//	float bp = dot(pos, bottomPlane.xyz);
//	float tp = dot(pos - topPlane.origin, topPlane.normal);
	float noise = 0;
	if (dot(vec4(pos, 1.0), bottomPlane) > 0 && dot(vec4(pos, 1.0), topPlane) > 0)
	{
		return 1.0;//cloudShape(pos);
	}
	return noise;
}

float StepRender(vec3 start, vec3 end)
{
	
	float sum = 1;
	int stepCount = 128;
	vec3 dir = normalize(end - start);
	float dis = length(end - start);
	float stepSize = dis / stepCount;
	vec3 samplePos = start;
	for (int i = 0; i < stepCount; i++)
	{
		float density = RenderShape(samplePos);
		if(density > 0)
			return 0.5;
		if (sum <= 0.01)
			break;
		//samplePos = start + dir * stepSize * (i + random(samplePos.xz));
		samplePos += dir * stepSize;
	}
	return sum;
}


float RenderBox(vec3 boundsMin, vec3 boundsMax, vec3 pos)
{
	float noise = 0;
	if (pos.x <boundsMax.x && pos.x >boundsMin.x &&
		pos.z <boundsMax.z && pos.z >boundsMin.z &&
		pos.y <boundsMax.y && pos.y >boundsMin.z)
	{
		return 1.0;
	}
	return noise;
}

float StepBox(vec3 boundsMin, vec3 boundsMax,vec3 start, vec3 end)
{
	float sum = 0.0;
	int stepCount = 128;
	vec3 dir = normalize(end - start);
	float dis = length(end - start);
	float stepSize = dis / stepCount;
	vec3 samplePos = start;
	for (int i = 0; i < stepCount; i++)
	{
		float density = RenderBox(boundsMin, boundsMax, samplePos);
		if(density > 0)
			return 0.5;
		if (sum <= 0.01)
			break;
		//samplePos = start + dir * stepSize * (i + random(samplePos.xz));
		samplePos += dir * stepSize;
	}
	return sum;
}

vec2 getPlaneUV(vec4 plane, vec3 pos)
{
	return vec2(pos.x, pos.z);
}


#define NOISE fbm
#define NUM_NOISE_OCTAVES 5

// Precision-adjusted variations of https://www.shadertoy.com/view/4djSRW
float hash(float p) { p = fract(p * 0.011); p *= p + 7.5; p *= p + p; return fract(p); }
float hash(vec2 p) {vec3 p3 = fract(vec3(p.xyx) * 0.13); p3 += dot(p3, p3.yzx + 3.333); return fract((p3.x + p3.y) * p3.z); }

float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}


float noise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

	// Four corners in 2D of a tile
	float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    // Simple 2D lerp using smoothstep envelope between the values.
	// return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
	//			mix(c, d, smoothstep(0.0, 1.0, f.x)),
	//			smoothstep(0.0, 1.0, f.y)));

	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}


float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}


float fbm(float x) {
	float v = 0.0;
	float a = 0.5;
	float shift = float(100);
	for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
		v += a * noise(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
}


//float fbm(vec2 x) {
//	float v = 0.0;
//	float a = 0.5;
//	vec2 shift = vec2(100);
//	// Rotate to reduce axial bias
//    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
//	for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
//		v += a * noise(x);
//		x = rot * x * 2.0 + shift;
//		a *= 0.5;
//	}
//	return v;
//}


float fbm(vec3 x) {
	float v = 0.0;
	float a = 0.5;
	vec3 shift = vec3(100);
	for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
		v += a * noise(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
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
	vec3 start = vec3(0,0,0);
	vec3 end = vec3(0,0,0);
	vec2 mapUv;
	bool isInsertStart = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 500.0,start,mapUv);
	bool isInsertEnd = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 600.0,end,mapUv);
//	if(isInsert)
//	{
//		
//	}
////	
//	bool isInsertStart = RayPlane(ubo.camPos, rayDir, bottomPlane,start);
	if(isInsertStart)
	{
		vec4 cloud = renderCloud(start, end, mapUv);
		float z = fbm(start);
		vec3 col =texture(weatherTexture1, vec2(start.x, start.y)).xyz;
		outColor = vec4(z); //texture(weatherTexture1, mapUv);//vec4(z);//vec4(col, length(col));
	}
	else
	{
		outColor =  vec4(0.0);
	}
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
