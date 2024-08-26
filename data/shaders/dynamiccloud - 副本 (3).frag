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

vec2 iResolution = vec2(0,0);


float treeLine = 0.0;
float treeCol = 0.0;


vec3 sunLight  = normalize( vec3(  0.4, 0.4,  0.48 ) );
vec3 sunColour = vec3(1.0, .9, .83);
float specular = 0.0;
vec3 cameraPos;
float ambient;
vec2 add = vec2(1.0, 0.0);
#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)

// This peturbs the fractal positions for each iteration down...
// Helps make nice twisted landscapes...
const mat2 rotate2D = mat2(1.3623, 1.7531, -1.7131, 1.4623);

// Alternative rotation:-
// const mat2 rotate2D = mat2(1.2323, 1.999231, -1.999231, 1.22);


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

float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( Hash12(p),          Hash12(p + add.xy),f.x),
                    mix( Hash12(p + add.yx), Hash12(p + add.xx),f.x),f.y);
    return res;
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
float Trees(vec2 p)
{
	
 	//return (texture(iChannel1,0.04*p).x * treeLine);
    return Noise(p*13.0)*treeLine;
}


//--------------------------------------------------------------------------
float FractalNoise(in vec2 xy)
{
	float w = .7;
	float f = 0.0;

	for (int i = 0; i < 4; i++)
	{
		f += Noise(xy) * w;
		w *= 0.5;
		xy *= 2.7;
	}
	return f;
}

//--------------------------------------------------------------------------
// Simply Perlin clouds that fade to the horizon...
// 200 units above the ground...
vec3 GetClouds(in vec3 sky, in vec3 rd)
{
	if (rd.y < 0.01) return sky;
	float v = -(200.0-cameraPos.y)/rd.y;
	rd.xz *= v;
	rd.xz += cameraPos.xz;
	rd.xz *= .010;
	float f = (FractalNoise(rd.xz) -.55) * 5.0;
	// Uses the ray's y component for horizon fade of fixed colour clouds...
	sky = mix(sky, vec3(.55, .55, .52), clamp(f*rd.y-.1, 0.0, 1.0));

	return sky;
}



//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
vec3 GetSky(in vec3 rd)
{
	float sunAmount = max( dot( rd, sunLight), 0.0 );
	float v = pow(1.0-max(rd.y,0.0),5.)*.5;
	vec3  sky = vec3(v*sunColour.x*0.4+0.18, v*sunColour.y*0.4+0.22, v*sunColour.z*0.4+.4);
	// Wide glare effect...
	sky = sky + sunColour * pow(sunAmount, 6.5)*.32;
	// Actual sun...
	sky = sky+ sunColour * min(pow(sunAmount, 1150.0), .3)*.65;
	return sky;
}

//--------------------------------------------------------------------------
// Merge mountains into the sky background for correct disappearance...
vec3 ApplyFog( in vec3  rgb, in float dis, in vec3 dir)
{
	float fogAmount = exp(-dis* 0.00005);
	return mix(GetSky(dir), rgb, fogAmount );
}

//--------------------------------------------------------------------------
// Calculate sun light...
void DoLighting(inout vec3 mat, in vec3 pos, in vec3 normal, in vec3 eyeDir, in float dis)
{
	float h = dot(sunLight,normal);
	float c = max(h, 0.0)+ambient;
	mat = mat * sunColour * c ;
	// Specular...
	if (h > 0.0)
	{
		vec3 R = reflect(sunLight, normal);
		float specAmount = pow( max(dot(R, normalize(eyeDir)), 0.0), 3.0)*specular;
		mat = mix(mat, sunColour, specAmount);
	}
}


//--------------------------------------------------------------------------
vec3 CameraPath( float t )
{
	float m = 0.0;//1.0+(iMouse.x/iResolution.x)*300.0;
	t = (cloudParams.time*1.5+m+657.0)*.006 + t;
    vec2 p = 476.0*vec2( sin(3.5*t), cos(1.5*t) );
	return vec3(35.0-p.x, -0.6, 4108.0+p.y);
}

//--------------------------------------------------------------------------
// Some would say, most of the magic is done in post! :D
vec3 PostEffects(vec3 rgb, vec2 uv)
{
	//#define CONTRAST 1.1
	//#define SATURATION 1.12
	//#define BRIGHTNESS 1.3
	//rgb = pow(abs(rgb), vec3(0.45));
	//rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	rgb = (1.0 - exp(-rgb * 6.0)) * 1.0024;
	//rgb = clamp(rgb+hash12(fragCoord.xy*rgb.r)*0.1, 0.0, 1.0);
	return rgb;
}

//--------------------------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord, vec3 raydir )
{
    vec2 xy = -1.0 + 2.0*fragCoord.xy / iResolution.xy;
	vec2 uv = xy * vec2(iResolution.x/iResolution.y,1.0);
	vec3 camTar;

//	#ifdef STEREO
//	float isCyan = mod(fragCoord.x + mod(fragCoord.y,2.0),2.0);
//	#endif

	cameraPos = ubo.camPos;
//	camTar.xyz	 = CameraPath(.1).xyz;
//	camTar.y = cameraPos.y = max(0.0, 1.5+sin(cloudParams.time*5.)*.5);
//    camTar.y -= smoothstep(60.0, 300.0,cameraPos.y)*150.;
	
//	float roll = 0.15*sin(cloudParams.time*.2);
//	vec3 cw = normalize(camTar-cameraPos);
//	vec3 cp = vec3(sin(roll), cos(roll),0.0);
//	vec3 cu = normalize(cross(cw,cp));
//	vec3 cv = normalize(cross(cu,cw));
	vec3 rd = raydir;//normalize( uv.x*cu + uv.y*cv + 1.5*cw );
	rd = normalize(rd);
//	#ifdef STEREO
//	cameraPos += .45*vec3(1,0,0)*isCyan; // move camera to the right - the rd vector is still good
//	#endif

	vec3 col;
	float distance;
	col = GetSky(rd);
	col = GetClouds(col, rd);

	col = PostEffects(col, uv);
	
//	#ifdef STEREO	
//	col *= vec3( isCyan, 1.0-isCyan, 1.0-isCyan );	
//	#endif
	
	fragColor=vec4(col,1.0);
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
	uv.x = uv.x + cloudParams.time;
	//uv.x = fract(uv.x / cloudParams.resolutionX) * cloudParams.resolutionX;

	uv.x = (uv.x / resolution.x) * 2 - 1;
	uv.y = (uv.y / resolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
	vec3 dirLength = (worldRayPos - ubo.camPos);
	vec3 rayDir = dirLength / length(dirLength);
	rayDir.y=-rayDir.y;

	vec3 start = vec3(0,0,0);
	vec3 end = vec3(0,0,0);
	vec2 mapUv;
	bool isInsertStart = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 500.0,start,mapUv);
	bool isInsertEnd = RayIntersectSphere(ubo.camPos, rayDir, vec3(0,0,0), 600.0,end,mapUv);

	vec4 fragColor;
	iResolution.x = cloudParams.resolutionX;
	iResolution.y = cloudParams.resolutionY;
	mainImage(fragColor,gl_FragCoord.xy, rayDir);
	outColor = fragColor;
}