#ifdef GL_ES
	precision highp float;
#endif

#define PI 3.14159265359
#define STEPS 256
#define EPS (2.0/iResolution.x)
#define FAR 6.0

layout(set = 0, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 model;
	mat4 view;
	vec3 camPos;
} ubo;

layout(set = 0, binding = 1) uniform Params
{
		vec4 lightDir;
		float exposure;
		float gamma;
		float prefilteredCubeMipLevels;
		float scaleIBLAmbient;
		float debugViewInputs;
		float debugViewEquation;
} lightParams;

layout(set = 0, binding = 2) uniform CloudParams{
	vec3 skyColor;
	float sunColourX;
	float sunColourY;
	float sunColourZ;
	float sunDirX;
	float sunDirY;
	float sunDirZ;
	float height;
	float resolutionX;
	float resolutionY;
	float time;
	float speed;
	int seed;
} cloudParams;

layout(location = 0) out vec4 outColor;

vec4 GetWorldPositionFromDepth(mat4 invVp, vec2 uv, float depth)
{
	vec4 wpos = invVp * vec4(uv, depth, 1.0);

	return wpos;
}

vec3 cameraPos;