// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#version 450

//#extension GL_KHR_vulkan_glsl : require
#extension GL_GOOGLE_include_directive : require

#include "includes/dynamiccommon.glsl"

float iTime = 0.0;
vec2 iResolution = vec2(.0);
vec3 iMouse = vec3(.0);
//Show shootiong star 
#define COMET

//varing light intensity for stars
#define FRACTAL_SKY
#define PULSED_STARS //for FRACTAL_SKY only

//for use in shadertoy environment or similia
#define SHADERTOY

#ifdef SHADERTOY
vec2 mouseR = vec2(0.);
vec2 mouseL = vec2(0.);
#else
uniform float time;
uniform vec2 mouseR;
uniform vec2 mouseL;
uniform vec2 resolution;
uniform sampler2D skyTex;
uniform sampler2D landTex;
#endif

const vec3 starColor = vec3(.43,.57,.97);

float contrast(float valImg, float contrast) { return clamp(contrast*(valImg-.5)+.5, 0., 1.); }
vec3  contrast(vec3 valImg, float contrast)  { return clamp(contrast*(valImg-.5)+.5, 0., 1.); }

float gammaCorrection(float imgVal, float gVal)  { return pow(imgVal, 1./gVal); }
vec3  gammaCorrection(vec3 imgVal, float gVal)   { return pow(imgVal, vec3(1./gVal)); }


//Get color luminance intensity
float cIntensity(vec3 col) { return dot(col, vec3(.299, .587, .114)); }


float hash( float n ) { return fract(sin(n)*758.5453); }
float rand(vec2 p) { return fract(sin(dot(p ,vec2(1552.9898,78.233))) * 43758.5453); }


float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    float n = p.x + p.y*57.0 + p.z*800.0;
    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x), mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
		    mix(mix( hash(n+800.0), hash(n+801.0),f.x), mix( hash(n+857.0), hash(n+858.0),f.x),f.y),f.z);
    return res;
}

float fbm( vec3 p )
{
    float f  = 0.50000*noise( p ); p *= 2.02;
          f += 0.25000*noise( p ); p *= 2.03;
          f += 0.12500*noise( p ); p *= 2.01;
          f += 0.06250*noise( p ); p *= 2.04;
          f += 0.03125*noise( p );
    return f*1.032258;
}


float fbm2( vec3 p )
{
    float f  = 0.50000*noise( p ); p *= 2.021;
          f += 0.25000*noise( p ); p *= 2.027;
          f += 0.12500*noise( p ); p *= 2.01;
          f += 0.06250*noise( p ); p *= 2.03;
          f += 0.03125*noise( p ); p *= 4.01;
          f += 0.015625*noise( p );p *= 8.04;
          f += 0.00753125*noise( p );
    return f*1.05;
}

float borealCloud(vec3 p)
{
	p+=fbm(vec3(p.x,p.y,0.0)*0.5)*2.25;
	float a = smoothstep(.0, .9, fbm(p*2.)*2.2-1.1);
	return a<0.0 ? 0.0 : a;
}

vec3 smoothCloud(vec3 c, vec2 pos)
{
	c*=0.75-length(pos-0.5)*0.75;
	float w=length(c);
	c=mix(c*vec3(1.0,1.2,1.6),vec3(w)*vec3(1.,1.2,1.),w*1.25-.25);
	return clamp(c,0.,1.);
}

float fractalField(in vec3 p,float s,  int idx) {
   float strength = 7. + .03 * log(1.e-6 + fract(sin(iTime) * 4373.11));
   float accum = s/4.;
   float prev = 0.;
   float tw = 0.;
   for (int i = 0; i < 24; ++i) {
      float mag = dot(p, p);
      p = abs(p) / mag + vec3(-.5, -.4, -1.5);
      float w = exp(-float(i) / 4.8);
      accum += w * exp(-strength * pow(abs(mag - prev), 2.7));
      tw += w;
      prev = mag;
   }
   return max(0., 5. * accum / tw - .7);
}

vec3 nrand3( vec2 co )
{
   vec3 a = fract( cos( co.x*8.3e-3 + co.y )*vec3(1.3e5, 4.7e5, 2.9e5) );
   vec3 b = fract( sin( co.x*0.3e-3 + co.y )*vec3(8.1e5, 1.0e5, 0.1e5) );
   vec3 c = mix(a, b, 0.5);
   return c;
}


#define iterations 16
#define formuparam 0.53 //77

#define volsteps 4
#define stepsize 0.00733

#define zoom   1.2700
#define tile   .850
#define speed  0.000

#define brightness 0.0007
#define darkmatter .1700
#define distfading 1.75
#define saturation .250



//besselham line function
//creates an oriented distance field from o to b and then applies a curve with smoothstep to sharpen it into a line
//p = point field, o = origin, b = bound, sw = StartingWidth, ew = EndingWidth, 
float shoothingStarLine(vec2 p, vec2 o, vec2 b, float sw, float ew){
	float d = distance(o, b);
	vec2  n = normalize(b - o);
	vec2 l = vec2( max(abs(dot(p - o, n.yx * vec2(-1.0, 1.0))), 0.0),
	               max(abs(dot(p - o, n) - d * 0.5) - d * 0.5, 0.0));
	return smoothstep( mix(sw, ew, 1.-distance(b,p)/d) , 0., l.x+l.y);
}

vec3 comet(vec2 p)
{
const float modu = 4.;        // Period: 4 | 8 | 16 
const float endPointY = -.1; // Hide point / Punto di sparizione Y
vec2 cmtVel = mod(iTime/modu+modu*.5, 2.) > 1. ? vec2(2., 1.4)*.5 : vec2(-2., 1.4)*.5;  // Speed component X,Y
vec2 cmtLen = vec2(.25)*cmtVel; //cmt lenght
    
    vec2 cmtPt = 1. - mod(iTime*cmtVel, modu);
    cmtPt.x +=1.;

    vec2 cmtStartPt, cmtEndPt;

    if(cmtPt.y < endPointY) {
        cmtEndPt   = cmtPt + cmtLen;
        if(cmtEndPt.y > endPointY) cmtStartPt = vec2(cmtPt.x + cmtLen.x*((endPointY - cmtPt.y)/cmtLen.y), endPointY);
        else                       return vec3(.0);
    }
    else {
        cmtStartPt = cmtPt;
        cmtEndPt = cmtStartPt+cmtLen; 
    }

    float bright = clamp(smoothstep(-.2,.65,distance(cmtStartPt, cmtEndPt)),0.,1.);

    vec2 dlt = vec2(.003) * cmtVel;

    float q = clamp( (p.y+.2)*2., 0., 1.);

    return  ( bright  * .75 *  (smoothstep(0.993, 0.999, 1. - length(p - cmtStartPt)) + shoothingStarLine(p, cmtStartPt, cmtStartPt+vec2(.06)*cmtVel,  0.009, 0.003)) +   //bulbo cmta
             vec3(1., .7, .2) * .33 * shoothingStarLine(p, cmtStartPt,         cmtEndPt,        0.003, .0003) +          // scia ...
             vec3(1., .5, .1) * .33 * shoothingStarLine(p, cmtStartPt+dlt,     cmtEndPt+dlt*2., 0.002 ,.0002) +         // ...
             vec3(1., .3, .0) * .33 * shoothingStarLine(p, cmtStartPt+dlt+dlt, cmtEndPt+dlt*4., 0.001, .0001)            //
            ) * (bright) * 
        q ; //attenuation on Y}
}


#define SMOOTH_V3(D,V,R)  { float p = D.y-.2; if(p < (V)) { float a = p/(V); R *= a *a ;  } }


vec3 smoothCloud3(vec3 c, vec3 pos)
{
	c*=0.75-length(pos-0.5)*0.75;
	float w=length(c);
	c=mix(c*vec3(1.0,1.2,1.6),vec3(w)*vec3(1.,1.2,1.),w*1.25-.25);
	return clamp(c,0.,1.);
}

vec3 renderBoreal(vec3 dir, float time)
{
    vec3 los = vec3(dir.x, dir.y, dir.z);

   float coord1 =borealCloud(dir * vec3(1.2,1.0,1.0) + vec3(0.0,time *0.22,0.0));
   float coord2 =borealCloud(dir * vec3(1.0,1.0,0.7) + vec3(0.0,time *0.27,0.0));
   float coord3 =borealCloud(dir * vec3(0.8,1.0,0.6) + vec3(0.0,time *0.29,0.0));
   float coord4 =borealCloud(dir * vec3(0.9,1.0,0.5) + vec3(0.0,time *0.20,0.0));
  
   vec3 boreal =  vec3(.1,1.,.5 ) * coord1 * 0.5 + vec3(.1,.9,.7) * coord2 * 0.9 + vec3(.75,.3,.99) * coord3 * 0.5 + vec3(.0,.99,.99) * coord4 * 0.57;


   SMOOTH_V3(dir,.5,boreal);
   SMOOTH_V3(dir,.35,boreal);
   SMOOTH_V3(dir,.27,boreal);

   smoothCloud3(boreal, dir); 
   boreal = gammaCorrection(boreal,1.3);
    
   float skyRange = max( dot( dir, vec3(0,1,0)), 0.0 );

   return boreal;
}
vec3 renderSky(vec3 rayDir, float time)
{
    vec3 skyPt = rayDir.xzy * 0.1 + vec3(-1.315, .39, 0.);
    vec3 freq = vec3(0.3, 0.67, 0.87);

    float ff = fractalField(skyPt,freq.z, 27);
    float skyRange = max( dot( rayDir, vec3(0,1,0)), 0.0 );
    ff  = min(pow(skyRange, 0.9), ff);

    vec3 sky =  vec3 (.75, 1., 1.4) * .01 * pow(2.4,ff*ff*ff)  * freq;

    return sky;// mix(sky, vec3(0.1,0.1,0.1), 0.10);
}
vec3 renderMoon(vec3 skyColor, vec3 rayDir, vec3 lightDir)
{
    vec3 moonColor =  vec3(1.0);//vec3(.99, .7, .8);
    float sunAmount = max( dot( rayDir, lightDir.xyz), 0.0 );
	float v = pow(1.0-max(rayDir.z,0.0),5.)*.5;
	vec3  sky = vec3(.0);//vec3(v*moonColor.x*0.1 + skyColor.x * 0.9, v*moonColor.y*0.1 + skyColor.y * 0.9, v*moonColor.z * 0.1 + skyColor.z * 0.9);
	sky = skyColor + moonColor * pow(sunAmount, 6.5) * 0.05;
	sky = sky + moonColor * min(pow(sunAmount, 1000.0), .5)*.5;
    return  sky;
}
vec3 renderStar(vec3 rayDir, float time)
{
   vec3 rnd = nrand3(rayDir.xy / rayDir.z * iResolution.x);
   float intensity = pow((1.+sin((iTime+27.)*rnd.x))*.5, 7.);
   float col = max(rnd.x * pow(rnd.y,7.) * intensity, 0.);

   float skyRange = max( dot( rayDir, vec3(0,1,0)), 0.0 );

   return vec3(min(pow(skyRange, 0.9), col) * col);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord, vec3 raydir )
{

#ifdef SHADERTOY
    mouseL = iMouse.xy / iResolution.xy;
#endif
    vec2 mouseLeft = vec2(0.0);
    vec2 mouse = mouseL;

	vec2 position = ( fragCoord / iResolution.xy );
    vec2 uv = 2. * position - 1.;
	uv.y = -uv.y;
    position.y+=0.2;

	vec2 coord= vec2((position.x-0.5)/position.y,1./(position.y+.2));

	float tm = iTime * .75 * .5;
	coord+=tm*0.0275 + vec2(1. - mouse.x, mouse.y);
	vec2 coord1=coord - tm*0.0275 + vec2(1. - mouse.x, mouse.y);

	
    vec2 ratio = iResolution.xy / max(iResolution.x, iResolution.y);;
    vec2 uvs = uv * ratio;

// Boreal effects
////////////////////////////////////
    // CloudColor * cloud * intensity
    vec3 boreal = vec3 (vec3(.1,1.,.5 )  * borealCloud(vec3(coord*vec2(1.2,1.), tm*0.22)) * .9  +
                        //vec3(.0,.7,.7 )  * borealCloud(vec3(coord1*vec2(.6,.6)  , tm*0.23)) * .5 +
                        vec3(.1,.9,.7) * borealCloud(vec3(coord*vec2(1.,.7)  , tm*0.27)) *  .9 +
                        vec3(.75,.3,.99) * borealCloud(vec3(coord1*vec2(.8,.6)  , tm*0.29)) *  .5 +
                        vec3(.0,.99,.99)  * borealCloud(vec3(coord1*vec2(.9,.5)  , tm*0.20)) *  .57);
                        


    boreal = smoothCloud(boreal, position);
    boreal = gammaCorrection(boreal,1.3);

// Sky background (fractal)
////////////////////////////////////
#ifdef FRACTAL_SKY
    // point position sky fractal    
    vec3 skyPt = vec3(uvs / 6., 0) + vec3(-1.315, .39, 0.);  //pt + pos
    
    skyPt.xy += vec2(- mouseLeft.x*.5*ratio.x, - mouseLeft.y * .5*ratio.y);

    //fractal
    vec3 freq = vec3(0.3, 0.67, 0.87);
    float ff = fractalField(skyPt,freq.z, 27);

    vec3 sky =  vec3 (.75, 1., 1.4) * .05 * pow(2.4,ff*ff*ff)  * freq ;
#else
    vec3 sky =  vec3 (0.);
#endif

// Moon
////////////////////////////////////
    vec2 moonPos = vec2(0.77,-.57) * ratio;
    float len = 1. - length((uvs - moonPos) ) ;
    // moon
    vec3 moon = vec3(.99, .7, .3) * clamp(smoothstep(0.95, 0.957, len) - 
                                          smoothstep(0.93, 0.957, 1. - length(uvs - (moonPos + vec2(0.006, 0.006)) ) - .0045)
                                          , 0., 1.) 

                                          * 2.;
    vec3 haloMoon  = vec3(.0,  .2,  .7)  * 0.2     * smoothstep(0.4, 0.9057, len);
         haloMoon += vec3(.5,  .5,  .85) * 0.0725  * smoothstep(0.7, 0.995,  len);

// Shooting star
////////////////////////////////////
#ifdef COMET
    vec3 cometA = comet(uvs);
#endif



// Horizon Light
////////////////////////////////////
         float pos = 1. - position.y;
         vec3 sunrise = vec3(.6,.3,.99) * (pos >  .45  ? (pos - .45)  * .65  : 0.) +  
                        vec3(.0,.7,.99) * (pos >= .585 ? (pos - .585  ) * .6  : 0.) +
                        vec3(.5,.99,.99)* (pos >= .67  ? (pos - .67) * .99 : 0.) ; 



// Intensity attenuation
////////////////////////////////////
         float ib = clamp(1.0-cIntensity(boreal)*3.,0.,1.);
         ib*=ib;
         float im = 1.0-cIntensity(moon);
         float ih = 1.0-cIntensity(haloMoon)*8.;
         float is = 1.0-cIntensity(sunrise)*6.;


         sky += ((moon + haloMoon)*ib*ib) * is * is +
                sunrise +
#ifdef COMET
                cometA +
#endif
                starColor  * ib *ib /*vec3(.3,.63,.97)*/  * im * im * ih * ih * is * is;

		 fragColor =  vec4(boreal + sky, 1.); 
}



void main()
{       
    cameraPos = ubo.camPos;
//    skyCol = 2.5*pow(cloudParams.skyColor, vec3(2.2));//cloudParams.skyColor;
    iResolution = vec2(cloudParams.resolutionX, cloudParams.resolutionY);
    iTime = cloudParams.seed +  cloudParams.time;
//    skyColor = cloudParams.skyColor;
//	sunColour = vec3(cloudParams.sunColourX, cloudParams.sunColourY, cloudParams.sunColourZ);
	mat4 matrix = ubo.projection * ubo.view;
	mat4 invMatrix= inverse(matrix);
	vec2 uv = gl_FragCoord.xy;

	uv.x = (uv.x / iResolution.x) * 2 - 1;
	uv.y = (uv.y / iResolution.y) * 2 - 1;

	vec3 worldRayPos = GetWorldPositionFromDepth(invMatrix, uv, 1.0).xyz;
    vec3 rayDir =  normalize(worldRayPos - vec3(0,0,0));
	vec4 fragColor = vec4(0);
	rayDir = vec3(rayDir.x, -rayDir.y, rayDir.z);
    mainImage(fragColor,gl_FragCoord.xy, rayDir);
    vec3 boreal = renderBoreal(rayDir,fract(iTime));
    vec3 sky = renderSky(rayDir,fract(iTime));
    vec3 moon =  renderMoon(sky, rayDir,vec3(0,1,0));
    vec3 star =  renderStar(rayDir,fract(iTime));
    float ib = clamp(1.0-cIntensity(boreal)*3.,0.,1.);
    ib*=ib;
    float im = 1.0-cIntensity(moon);

    sky += ((moon)*ib*ib) +star + boreal;

	outColor = vec4(vec3(borealCloud(rayDir)), 1.0);//fragColor;
}

