#version 450

//#extension GL_KHR_vulkan_glsl : require
#extension GL_GOOGLE_include_directive : require

#include "includes/dynamiccommon.glsl"

#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)

vec2 iResolution = vec2(0,0);
int cloudType = 1;
float cloudscale = 0.01;
vec3 sunColour = vec3(0.9,0.81,0.71);//vec3(1.0, .85, .78);
vec3 skyColor = vec3(0.18,0.22,0.4);
float specular = 0.0;

vec2 add = vec2(1.0, 0.0);


float seed1 = 43758.5453123;
const mat2 rotate2D = mat2(1.3623, 1.7531, -1.7131, 1.4623);


vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*seed1);
}
//  1 out, 2 in...
float Hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 Hash22(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return fract((p3.xx+p3.yz)*p3.zy);

}
float Hash31 (in vec2 st) {
    float key = dot(st.xy, vec2(12.9898,78.233));
    float kv = sin(key) * seed1; //不能搞太大 会溢出
    return fract(kv);
}
float baseNoise1( in vec2 st ) {
    vec2 p = floor(st);
    vec2 f = fract(st);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix( Hash12(p),          Hash12(p + add.xy),f.x),
                    mix( Hash12(p + add.yx), Hash12(p + add.xx),f.x),f.y);
    return res;
}

float baseNoise2( in vec2 st )
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = Hash31(i);
    float b = Hash31(i + vec2(1.0, 0.0));
    float c = Hash31(i + vec2(0.0, 1.0));
    float d = Hash31(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}
float baseNoise3 (vec2 st) {

    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(st + (st.x+st.y)*K1);	
    vec2 a = st - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	

}


float Noise( in vec2 x )
{
    return baseNoise1(x);
}

vec2 Noise2( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y * 57.0;
   vec2 res = mix(mix( Hash22(p),          Hash22(p + add.xy),f.x),
                  mix( Hash22(p + add.yx), Hash22(p + add.xx),f.x),f.y);
    return res;
}


//--------------------------------------------------------------------------
float FractalNoise1(in vec2 xy)
{

    // Initial values
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 3.0;

	float total = 0.0;
	for (int i = 0; i < 6; i++) {
		total += baseNoise1(xy) * amplitude;
		amplitude *= 0.5;
		xy *= frequency;
	}
	return total;
}
float FractalNoise2 (in vec2 xy) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 0.0;
    for (int i = 0; i < 6; i++) {
        value += amplitude * baseNoise2(xy);
        xy *= 3.0;
        amplitude *= .5;
    }
    return value;
}

const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
float FractalNoise3(vec2 xy) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += baseNoise3(xy) * amplitude;
		xy = m * xy;
		amplitude *= 0.4;
	}
	return total;
}

//加细节
vec3 GetClouds(in vec3 sky, in vec3 rd)
{
	if (rd.y < 0.01) return sky;
	float v = -(cloudParams.height-cameraPos.y)/rd.y;
	rd.xz *= v;
	rd.xz += cameraPos.xz;
	rd.xz *= .010;
	float f =0.0;
    vec2 mapUv = rd.xz;
    vec2 uv = mapUv;
    float time = cloudParams.time;
    float speed = cloudParams.speed;
    float q = 0;
	 time = time * speed * 2.0;
	uv -= - time;
	if(cloudType == 1)
	{
		q = FractalNoise1(uv * cloudscale * 0.5); 
	}
	else if(cloudType == 2)
	{
		q = FractalNoise2(uv * cloudscale * 0.5); 
	}
	else
	{	
		q = FractalNoise3(uv * cloudscale * 0.5); 
	}
    //ridged noise shape
	float r = 0.0;
	uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*baseNoise3( uv )); // 1 2 3
        uv = m*uv + time;
		weight *= 0.7;
    }
    
    //noise shape
    uv = mapUv;
	uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*baseNoise3( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    //noise colour
    float c = 0.0;
    time = time * speed * 2.0;
    uv = mapUv;
	uv *= cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*baseNoise3( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    //noise ridge colour
    float c1 = 0.0;
    time = time * speed * 3.0;
    uv = mapUv;
	uv *= cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*baseNoise3( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c +=c1;
    f = f*r;
    f +=c;
	sky = mix(sky, vec3(.55, .55, .55), clamp(f* rd.y -.1, 0.0, 1.0));

	return sky;
}


//--------------------------------------------------------------------------
vec3 GetSky(in vec3 rd)
{

	float sunAmount = max( dot( rd, lightParams.lightDir.xyz), 0.0 );
	float v = pow(1.0-max(rd.y,0.0),5.)*.5;
	vec3  sky = vec3(v*sunColour.x*0.3 + skyColor.x * 0.7, v*sunColour.y*0.3 + skyColor.y * 0.7, v*sunColour.z * 0.3 + skyColor.z * 0.7);
	sky = sky * 0.8 + sunColour * pow(sunAmount, 6.5)*.2;
	sky = sky * 0.6 + sunColour * min(pow(sunAmount, 5000.0), .3)*.4;
	return sky;
}

//--------------------------------------------------------------------------
vec3 ApplyFog( in vec3  rgb, in float dis, in vec3 dir)
{
	float fogAmount = exp(-dis* 0.00005);
	return mix(GetSky(dir), rgb, fogAmount );
}



//--------------------------------------------------------------------------
vec3 PostEffects(vec3 rgb, vec2 uv)
{
	
	rgb = (1.0 - exp(-rgb * 6.0)) * 1.0024;
	return rgb;
}

//--------------------------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord, vec3 raydir )
{
    vec2 xy = -1.0 + 2.0*fragCoord.xy / iResolution.xy;
	vec2 uv = xy * vec2(iResolution.x/iResolution.y,1.0);
	vec3 camTar;

	cameraPos = ubo.camPos;

	vec3 rd = raydir;
	rd = normalize(rd);

	vec3 col;
	col = GetSky(rd);
	col = GetClouds(col, rd);
	col = PostEffects(col, uv);
	
	fragColor=vec4(col,1.0);
}



void main()
{
    vec2 resolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
	skyColor = cloudParams.skyColor;
	sunColour = vec3(cloudParams.sunColourX, cloudParams.sunColourY, cloudParams.sunColourZ);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
	vec2 uv = gl_FragCoord.xy;
	uv.x = uv.x + cloudParams.time;

	uv.x = (uv.x / resolution.x) * 2 - 1;
	uv.y = (uv.y / resolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;

	vec3 rayDir =  normalize(worldRayPos - vec3(0,0,0));

	rayDir = vec3(rayDir.x, -rayDir.y, rayDir.z);

	seed1 = cloudParams.seed % 1000 * 10000 + 40000;
	cloudType = cloudParams.seed % 3 + 1;
	cloudscale = cloudParams.seed % 10 * 0.001 + 0.001;//0.01-0.001
	vec4 fragColor = vec4(0);
	iResolution.x = cloudParams.resolutionX;
	iResolution.y = cloudParams.resolutionY;
	mainImage(fragColor,gl_FragCoord.xy, rayDir);
	outColor = fragColor;//vec4(GetClouds(vec3(0.0,.0,.0), rayDir), 1.0); //fragColor;
}