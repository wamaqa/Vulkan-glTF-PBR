// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#version 450

//#extension GL_KHR_vulkan_glsl : require
#extension GL_GOOGLE_include_directive : require

#include "includes/dynamiccommon.glsl"

float iTime = 0.0;
vec2 iResolution = vec2(.0);
vec3 iMouse = vec3(.0);
#define period 15.
//#define fxrand floor(iTime/period)+iDate.z
#define fxrand floor(iTime/period)
#define hash1 rnd(fxrand)
#define hash2 rnd(fxrand+.111)
#define hash3 rnd(fxrand+.222)
#define hash4 rnd(fxrand+.333)
#define hash5 rnd(fxrand+.444)
#define hash6 rnd(fxrand+.555)
#define hash7 rnd(fxrand+.666)
#define hash8 rnd(fxrand+.777)
#define hash9 rnd(fxrand+.877)
#define hash10 rnd(fxrand+.997)
#define hash11 rnd(fxrand+1.11777)
#define hash12 rnd(fxrand+1.411777)

#define STEPS 256
#define EPS (2.0/iResolution.x)
#define FAR 6.0
#define PI 3.14159265359

#define iGT (iTime+10.0)
mat2 m = mat2(0.8,-0.6,0.6,0.8);

//const mat2 m = mat2( 1.6, .2, -1.2,  1.6 );
vec2 uvd;
float zoom;

float hashh(vec2 p) {
	return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

float hash(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 x) {
	vec2 i = floor(x);
	vec2 f = fract(x);
	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (in vec2 p) {

    float value = 0.0;
    float freq = 1.0;
    float amp = .5;

    for (int i = 0; i < 14; i++) {
        value += amp * (noise((p - vec2(1.0)) * freq));
        freq *= 1.9;
        amp *= 0.6;
    }
    return value;
}


mat2 rot(float a)
{
    float s=sin(a), c=cos(a);
    return mat2(c,s,-s,c);
}

float rnd(float p)
{
    p*=1234.5678;
    p = fract(p * .1031);
    p *= p + 33.33;
    return fract(2.*p*p);
}


float rand(float r){
	vec2	co=vec2(cos(r*428.7895),sin(r*722.564));
	return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}


vec3 render(vec3 dir)
{
	float s=0.3,fade=1., fade2=1., pa=0., sd=0.2;
	vec3 v=vec3(0.);
    dir.y+=4.*hash4;
    dir.x+=hash5;
	for (float r=0.; r<15.; r++) {
		vec3 p=s*dir;
        mat2 rt=rot(r);
        p.xz*=rt;
        p.xy*=rt;
        p.yz*=rt;
		p = abs(1.-mod(p*(hash1*2.+1.),2.));
		float pa,a=pa=0.;
		for (int i=0; i<13; i++) {
			if (float(i)>mod(iTime,period)*10.) break;
			p=abs(p)/dot(p,p)-.7-step(.5,hash10)*.1;
			float l=length(p)*.5;
			a+=abs(l-pa);
			pa=length(p);
		}
        fade*=.96;
		sd+=.5;
		float cv=abs(2.-mod(sd,4.));
		v+=normalize(vec3(cv*2.,cv*cv,cv*cv*cv))*pow(a*.02,2.)*fade;
		v.rb*=rot(hash3*3.);
        v=abs(v);
		pa=a;
		s+=.05;
	}
	float sta=v.x;
	vec3 roj=vec3(1.5,1.,.8);
	uvd.x*=sign(hash12-.5);
	uvd*=rot(radians(360.*hash8));
	uvd.y*=1.+(uvd.x+.5)*1.;
	v=pow(v,1.-.5*vec3(smoothstep(.5,0.,abs(uvd.y))));
	v+=.04/(.1+abs(uvd.y*uvd.y))*roj*min(1.,iTime*.3);
	float core=smoothstep(.3,0.,length(uvd))*1.2*min(1.,iTime*.3);
	v+=core*roj;
	v=mix(vec3(length(v)*.7),v,.45);
	float neb=fbm(dir.xy*15.)-.5;
	uvd.y+=neb*.3;
	neb=pow(smoothstep(.8,.0,abs(uvd.y)),2.)*.9;
	v=mix(v*vec3(1.,.9,1.2),vec3(0.),max(neb,.7-neb)+core*.06-sta*.1);
	return pow(v,vec3(1.05))*1.2;
}



float Noise2d( in vec2 x )
{
    float xhash = cos( x.x * 37.0 );
    float yhash = cos( x.y * 57.0 );
    return fract( 415.92653 * ( xhash + yhash ) );
}
vec3 noisen(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Quintic because everyone else is using the cubic! :D
    vec2 df = 30.0*f*f*(f*(f-2.0)+1.0);
    f = f*f*f*(f*(f*6.-15.)+10.);
    
    float a = Noise2d(i+vec2(0.5, 0.5));
    float b = Noise2d(i+vec2(1.5, 0.5));
    float c = Noise2d(i+vec2(0.5, 1.5));
    float d = Noise2d(i+vec2(1.5, 1.5));
    
    float k = a-b-c+d;
    float n = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    
    return vec3(n, vec2(b-a+k*f.y, c-a+k*f.x)*df);
}

float fbmSimple(vec2 p) {
    float f = 0.0;
    f += 0.5*noisen(p).x; p = 2.0*m*p;
    f += 0.25*noisen(p).x; p = 2.0*m*p;
    f += 0.125*noisen(p).x; p = 2.0*m*p;
    f += 0.0625*noisen(p).x;
    return f/0.9375;
}

float fbmL(vec2 p) {
    vec2 df = vec2(0.0);
    float f = 0.0, w = 0.5;
    
    for (int i = 0; i < 2; ++i) {
        vec3 n = noisen(p);
        df += n.yz;
        f += abs(w * n.x/ (1.0 + dot(df, df)));
        w *= 0.5; p = 2.*m*p;
    }
    return f;
}

float fbmM(vec2 p) {
    vec2 df = vec2(0.0);
    float f = 0.0, w = 0.5;
    
    for (int i = 0; i < 4; ++i) {
        vec3 n = noisen(p);
        df += n.yz;
        f += abs(w * n.x/ (1.0 + dot(df, df)));
        w *= 0.5; p = 2.*m*p;
    }
    return f;
}

float fbmH(vec2 p) {
    vec2 df = vec2(0.0);
    float f = 0.0, w = 0.5;
    
    for (int i = 0; i < 10; ++i) {
        vec3 n = noisen(p);
        df += n.yz;
        f += abs(w * n.x/ (1.0 + dot(df, df)));
        w *= 0.5; p = 2.*m*p;
    }
    return f;
}

float map(vec3 p) {
    return p.y - fbmM(p.xz);
}

float mapL(vec3 p) {
    return p.y - fbmL(p.xz);
}

float mapH(vec3 p) {
    return p.y - fbmH(p.xz);
}

vec3 normal(vec3 p) {
    vec2 q = vec2(0., EPS);
    return normalize(vec3(mapH(p+q.yxx) - mapH(p-q.yxx),
                		  mapH(p+q.xyx) - mapH(p-q.xyx),
                		  mapH(p+q.xxy) - mapH(p-q.xxy)));
}

vec3 normalL(vec3 p) {
    vec2 q = vec2(0., EPS);
    return normalize(vec3(mapL(p+q.yxx) - mapL(p-q.yxx),
                		  mapL(p+q.xyx) - mapL(p-q.xyx),
                		  mapL(p+q.xxy) - mapL(p-q.xxy)));
}

vec3 skyCol = 2.5*pow(vec3(40., 56., 84.)/255., vec3(2.2));
vec3 moonCol = pow(vec3(168., 195., 224.)/255., vec3(2.2));

vec3 fogColor(vec3 sundir, vec3 dir) {
    vec3 col = skyCol + moonCol*pow(max(dot(sundir, dir), 0.0), 16.0)*max(0.0, -dir.z);
    return col / (col + 1.0);
}

vec3 fullSky(vec3 sundir, vec3 dir) {
    vec3 stars = vec3(smoothstep(0.8, 0.95, fbmSimple(100.0*dir.xy/(dir.z+EPS))));
    
    vec3 clouds = vec3(0.0);
    float s = 0.25;
    for (int i = 0; i < 3; ++i) {
    	clouds += fbmSimple(dir.xz/(dir.y+EPS)-s*iGT);
        s *= 1.35;
    }
    
    vec3 col = skyCol + 0.15*clouds*max(0.0, dir.y);
    col += 2.0*stars*max(0.0, dir.y);
    
    col += max(0.0, -dir.z)*moonCol*pow(max(dot(sundir, dir), 0.0), 16.0);
    vec2 moonPos = dir.xy/dir.z - sundir.xy/sundir.z;
    col = mix(col, vec3(1.65), max(0.0, -dir.z)*fbmSimple(8.5*moonPos)*smoothstep(0.37, 0.35, length(moonPos)));
    
    return col / (col + 1.0);
}

vec3 material(vec3 p, vec3 n) {
    vec3 brown = pow(vec3(185., 122., 87.)/255., vec3(2.2));
    return mix(vec3(1.0), brown, smoothstep(0.6*n.y, 1.0*n.y, fbmH(p.xz)));
}

vec3 shade(vec3 ro, vec3 rd, float t) {
    vec3 l = normalize(vec3(1.0, 0.0, -1.0));
      
    vec3 col = fullSky(l, rd);
    return clamp(col / (col + 1.0), 0.0, 1.0);
}

float raymarch(vec3 ro, vec3 rd) {
    float d = 0., t = 0.0,eps=EPS;
    for (int i = 0; i < STEPS; ++i) {
        d = map(ro + t*rd);
        if (d < eps*t || t > FAR)
            break;
        t += max(0.35*d, 2.*EPS*t);
        eps *= 1.1;
    }
   
    return d < eps*t ? t : -1.;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord , vec3 raydir )
{
	vec2 uv = (-iResolution.xy + 2.0*fragCoord.xy) / iResolution.y;
//    
//    vec3 ro = vec3(0.0, 0.0, -0.2*iGT); ro.y = 0.15 + 1.2*fbmL(ro.xz);
//    vec3 rd = normalize(vec3(uv, -1.0));
    
    float t = raymarch(cameraPos, raydir);
    vec3 col = pow(shade(cameraPos, raydir, t), vec3(1.0/2.2));
    
    col = smoothstep(0.0, 1.0, col);
    col *= 1.2;

	uvd=raydir.xz;///raydir.z;

//    col =col + render(raydir);
    fragColor = vec4(col,1.0);
}

void main()
{       
    cameraPos = ubo.camPos;
    skyCol = 2.5*pow(cloudParams.skyColor, vec3(2.2));//cloudParams.skyColor;
    iResolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
    iTime = cloudParams.seed +  cloudParams.time;
//    skyColor = cloudParams.skyColor;
//	sunColour = vec3(cloudParams.sunColourX, cloudParams.sunColourY, cloudParams.sunColourZ);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
	vec2 uv = gl_FragCoord.xy;
//	uv.x = uv.x + cloudParams.time;

	uv.x = (uv.x / iResolution.x) * 2 - 1;
	uv.y = (uv.y / iResolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
    vec3 rayDir =  normalize(worldRayPos - vec3(0,0,0));
	vec4 fragColor = vec4(0);
	rayDir = vec3(rayDir.x, -rayDir.y, rayDir.z);
    mainImage(fragColor,gl_FragCoord.xy, rayDir);
	outColor = fragColor;
}

