// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#version 450

//#extension GL_KHR_vulkan_glsl : require
#extension GL_GOOGLE_include_directive : require

#include "includes/dynamiccommon.glsl"

float iTime = 0.0;
vec2 iResolution = vec2(.0);
vec3 iMouse = vec3(.0);



//Show shootiong star 
mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}
mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);
float tri(in float x){return clamp(abs(fract(x)-.5),0.01,0.49);}
vec2 tri2(in vec2 p){return vec2(tri(p.x)+tri(p.y),tri(p.y+tri(p.x)));}

#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)

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
	float v = -(cloudParams.height * 0.1-cameraPos.y)/rd.y;
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
vec3 PostEffects(vec3 rgb)
{
	
	rgb = (1.0 - exp(-rgb * 10.0)) * 1.0024;
	return rgb;
}

float triNoise2d(in vec2 p, float spd)
{
    return FractalNoise1(p * 4);
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
    return  clamp(1./pow(rz*29., 1.3),0.,.55);
}

float hash21(in vec2 n){ return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453); }
vec4 aurora(vec3 ro, vec3 rd)
{
    vec4 col = vec4(0);
    vec4 avgCol = vec4(0);
    
    for(float i=0.;i<50.;i++)
    {
        float of = 0.006*hash21(gl_FragCoord.xy)*smoothstep(0.,15., i);
        float pt = ((.8+pow(i,1.4)*.002)-ro.y)/(rd.y*2.+0.4);
        pt -= of;
    	vec3 bpos = ro + pt*rd;
        vec2 p = bpos.zx;
        float rzt = triNoise2d(p, 0.06);
        vec4 col2 = vec4(0,0,0, rzt);
        col2.rgb = (sin(1.-vec3(2.15,-.5, 1.2)+i*0.043)*0.5+0.5)*rzt;
        avgCol =  mix(avgCol, col2, .5);
        col += avgCol*exp2(-i*0.065 - 2.5)*smoothstep(0.,5., i);
        
    }
    
    col *= (clamp(rd.y*15.+.4,0.,1.));
    
    
    //return clamp(pow(col,vec4(1.3))*1.5,0.,1.);
    //return clamp(pow(col,vec4(1.7))*2.,0.,1.);
    //return clamp(pow(col,vec4(1.5))*2.5,0.,1.);
    //return clamp(pow(col,vec4(1.8))*1.5,0.,1.);
    
    //return smoothstep(0.,1.1,pow(col,vec4(1.))*1.5);
    return col*1.8;
    //return pow(col,vec4(1.))*2.
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
        vec2 rn = nmzHash33(id).xy;
        float c2 = 1.-smoothstep(0.,.6,length(q));
        c2 *= step(rn.x,.0005+i*i*0.001);
        c += c2*(mix(vec3(1.0,0.49,0.1),vec3(0.75,0.9,1.),rn.y)*0.1+0.9);
        p *= 1.3;
    }
    return c*c*.8;
}

vec3 bg(in vec3 rd)
{
    float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd)*0.5+0.5;
    sd = pow(sd, 5.);
    vec3 col = mix(vec3(0.05,0.1,0.2), vec3(0.1,0.05,0.2), sd);
    return col*.63;
}
//-----------------------------------------------------------
vec3 renderMoon(vec3 skyColor, vec3 rayDir, vec3 lightDir)
{
    vec3 moonColor =  vec3(1.0);//vec3(.99, .7, .8);
    float sunAmount = max( dot( rayDir, lightDir.xyz), 0.0 );
	float v = pow(1.0-max(rayDir.z,0.0),5.)*.5;
	vec3  sky = skyColor + moonColor * pow(sunAmount, 6.5) * 0.05;
	sky = sky + moonColor * min(pow(sunAmount, 1000.0), .5)*.5;
    return  sky;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord , vec3 raydir )
{
    vec3 ro = vec3(0,0,0);
    vec3 rd = raydir;
    vec3 col = vec3(0.);
    vec3 brd = rd;

   vec3 cloudCol = GetClouds(vec3(0.,5.,1.0), raydir);
    cloudCol = PostEffects(cloudCol);
    vec4 cloud = vec4(cloudCol, length(cloudCol));

    float fade = smoothstep(0.,0.01,abs(brd.y))*0.1+0.9;
    
    
    col = bg(rd)*fade;
    
    if (rd.y > 0.){
        vec4 aur = smoothstep(0.,1.5,aurora(ro,rd))*fade;
        col += stars(rd);
        col = col*(1.-aur.a) + aur.rgb;
    }
    else //Reflections
    {
        rd.y = abs(rd.y);
        col = bg(rd)*fade*0.6;
        vec4 aur = smoothstep(0.0,2.5,aurora(ro,rd));
        col += stars(rd)*0.1;
        col = col*(1.-aur.a) + aur.rgb;
        vec3 pos = ro + ((0.5-ro.y)/rd.y)*rd;
        float nz2 = triNoise2d(pos.xz*vec2(.5,.7), 0.);
        col += mix(vec3(0.2,0.25,0.5)*0.08,vec3(0.3,0.3,0.5)*0.7, nz2*0.4);
    }
    
	fragColor = vec4(cloudCol, 1.);
}

vec3 getSky(in vec3 rd)
{
    float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd)*0.5+0.5;
    sd = pow(sd, 5.);
    vec3 col = mix(vec3(0.05,0.1,0.2), vec3(0.1,0.05,0.2), sd);
    return col*.63;
}

void main()
{       
    cameraPos = ubo.camPos;
    iResolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
    iTime = cloudParams.time * 50;
    vec3 lightDir = vec3(0.5,0.5,0.5);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
	vec2 uv = gl_FragCoord.xy;

	uv.x = (uv.x / iResolution.x) * 2 - 1;
	uv.y = (uv.y / iResolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
    vec3 rayDir =  normalize(worldRayPos - vec3(0,0,0));
	vec4 fragColor = vec4(0);
	rayDir = vec3(rayDir.x, -rayDir.y, rayDir.z);
    
    vec3 col = getSky(rayDir);
    col += stars(rayDir);
    col += renderMoon(col, rayDir,lightDir);
    col += GetClouds(col, rayDir);
//    mainImage(fragColor,gl_FragCoord.xy, rayDir);


	outColor =vec4(col, 1.0); fragColor;
}

