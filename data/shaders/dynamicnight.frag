// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#version 450

//#extension GL_KHR_vulkan_glsl : require
#extension GL_GOOGLE_include_directive : require

#include "includes/dynamiccommon.glsl"

#define saturate(x) clamp(x, 0.0, 1.0)

vec2 iResolution = vec2(0.0);
float iTime = 0.0;
vec3 _skyColor = vec3(0.0);
vec3 _sunColor  = vec3(0.0);
vec3 _sunDir  = vec3(0.0);
float _cloudscale = 1.0;
float seed1 = 43758.5453123;
#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)

mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}
mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);
float tri(in float x){return clamp(abs(fract(x)-.5),0.01,0.49);}
vec2 tri2(in vec2 p){return vec2(tri(p.x)+tri(p.y),tri(p.y+tri(p.x)));}

float triNoise2d(in vec2 p, float spd)
{
    float z=1.8;
    float z2=2.5;
	float rz = 0.;
    p *= mm2(p.x*0.06);
    vec2 bp = p;
	for (float i=0.; i<5.; i++ )
	{
        vec2 dg = tri2(bp*1.85)*.75;
        dg *= mm2(iTime*spd);
        p -= dg/z2;

        bp *= 1.3;
        z2 *= .45;
        z *= .42;
		p *= 1.21 + (rz-1.0)*.02;
        
        rz += tri(p.x+tri(p.y))*z;
        p*= -m2;
	}
    return clamp(1./pow(rz*29., 1.3),0.,.55);
}

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


float hash21(in vec2 n){ return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453); }
vec2 add = vec2(1.0, 0.0);
float baseNoise( in vec2 st ) {
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
const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

float FractalNoise(in vec2 xy)
{

	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += baseNoise(xy) * amplitude;
		xy = m * xy;
		amplitude *= 0.4;
	}
	return total;
}
vec3 renderCloudFog(vec3 sky, vec3 rd)
{
    if (rd.y < 0.01) return sky;
    vec3 col = vec3(0);
    vec3 avgCol = vec3(0);
//    if (rd.y < 0.01) return sky;
	float v = -(cloudParams.height-cameraPos.y)/rd.y;
	rd.xz *= v;
//	rd.xz += cameraPos.xz;
	rd.xz *= .010;
//	float f =0.0;
    vec2 mapUv = rd.xz * 10;
    vec2 uv = mapUv;
    float time = cloudParams.time;
    float speed = cloudParams.speed;
    time = cloudParams.time * cloudParams.speed *10000;
    _cloudscale = _cloudscale * 0.1;
    uv += time;
    float q = FractalNoise(uv * _cloudscale);
   
    q = clamp(abs(q)* rd.y - .1, 0.0, 1.0);
    sky = mix(sky, vec3(.9, .9, .9), q);
    return sky;
}

//加细节
vec3 renderCloud(in vec3 sky, in vec3 rd)
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
	q = FractalNoise(uv * _cloudscale * 0.5); 

    //ridged noise shape
	float r = 0.0;
	uv *= _cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*baseNoise( uv )); // 1 2 3
        uv = m*uv + time;
		weight *= 0.7;
    }
    
    //noise shape
    uv = mapUv;
	uv *= _cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*baseNoise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    //noise colour
    float c = 0.0;
    time = time * speed * 2.0;
    uv = mapUv;
	uv *= _cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*baseNoise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    //noise ridge colour
    float c1 = 0.0;
    time = time * speed * 3.0;
    uv = mapUv;
	uv *= _cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*baseNoise( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c +=c1;
    f = f*r;
    f +=c;
	sky = mix(sky, vec3(.55, .55, .55), clamp(f* rd.y -.1, 0.0, 1.0));

	return sky;
}


//-------------------Background and Stars--------------------

vec3 nmzHash33(vec3 q)
{
    uvec3 p = uvec3(ivec3(q));
    p = p*uvec3(374761393U, 1103515245U, 668265263U) + p.zxy + p.yzx;
    p = p.yzx*(p.zxy^(p >> 3U));
    return vec3(p^(p >> 16U))*(1.0/vec3(0xffffffffU));
}

vec3 stars(in vec3 p)
{
    vec3 c = vec3(0.);
    float res = iResolution.x*1.;
    
	for (float i=0.;i<4.;i++)
    {
        vec3 q = fract(p*(.15*res))-0.5;
        vec3 id = floor(p*(.15*res));
        vec2 rn = nmzHash33(id).xy * 10;
        float c2 = 1.-smoothstep(0.,.6,length(q));
        c2 *= step(rn.x,.0005+i*i*0.001);
        c += c2*(mix(vec3(1.0,0.49,0.1),vec3(0.75,0.9,1.),rn.y)*0.01+0.9);
        p *= 1.3;
    }
    return c*c*.8;
}

vec3 renderSky(in vec3 rd)
{
    float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd)*0.5+0.5;
    sd = pow(sd, 5.);
    vec3 col = mix(vec3(0.05,0.1,0.2), vec3(0.1,0.05,0.2), sd);
    return _skyColor;//vec3(0.05,0.1,0.2)*.63;
}
//-----------------------------------------------------------
vec3 renderMoon(vec3 skyColor, vec3 rayDir)
{
    vec3 moonColor =  _sunColor;
    float sunAmount = max( dot( rayDir, _sunDir), 0.0 );
	float v = pow(1.0-max(rayDir.z,0.0),5.)*.5;
	vec3  sky = skyColor + moonColor * pow(sunAmount, 6.5) * 0.05;
	sky = sky + moonColor * min(pow(sunAmount, 8000.0), .15)*.5;
    return  sky;
}
vec3 PostEffects(vec3 rgb)
{
	rgb = (1.0 - exp(-rgb * 0.2)) * 1.0024;
	return rgb;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord , vec3 raydir )
{
    vec3 ro = vec3(0,0,0);
    vec3 rd = raydir;
    vec3 col = vec3(0.);
    vec3 brd = rd;
    float fade = smoothstep(0.,0.05,abs(brd.y))*0.1+0.9;
    
    col = renderSky(rd)*fade;
    col = renderCloud(col, rd);
    col = PostEffects(col);
    vec3 moon = renderMoon(col, raydir);
    if(raydir.y >0)
    {
        moon += stars(raydir);
    }
	fragColor = vec4(moon, 1.);
}

void main()
{       
    cameraPos = vec3(0.0);//ubo.camPos;
    iResolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
    iTime = cloudParams.time;
    _skyColor = cloudParams.skyColor;
    _sunColor = vec3(cloudParams.sunColourX, cloudParams.sunColourY, cloudParams.sunColourZ);
    _sunDir = normalize(vec3(cloudParams.sunDirX, cloudParams.sunDirY, cloudParams.sunDirZ));
    _cloudscale = cloudParams.seed % 10 * 0.001 + 0.001;
    vec3 lightDir = vec3(0.5,0.5,0.5);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
    vec2 fragCoord = gl_FragCoord.xy;
	vec2 uv = gl_FragCoord.xy;

	uv.x = (uv.x / iResolution.x) * 2 - 1;
	uv.y = (uv.y / iResolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
    vec3 rayDir =  normalize(worldRayPos -cameraPos);
	vec4 fragColor = vec4(0);
	rayDir = vec3(rayDir.x, -rayDir.y, rayDir.z);
    
    mainImage(fragColor, fragCoord, rayDir);
    outColor = fragColor;
}

