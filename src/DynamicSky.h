#ifndef __DYCLOUD__
#define __DYCLOUD__
#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <iostream>
#include <VulkanSwapChain.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include "VulkanDevice.hpp"
#include "VulkanglTFModel.h"
#include <VulkanTexture.hpp>
class VulkanDevice;
#define AUTOUPDATEShader


struct CloudParams
{
	/// <summary>
	/// 云层高度
	/// </summary>
	float height;
	float resolutionX;
	float resolutionY;
	std::string dataPath;
};

struct CloudUBO
{
	glm::vec3 skyColor = glm::vec3(0.081f, 0.151f, 0.488f);
	glm::vec3 sunColor = glm::vec3(0.9f, 0.81f, 0.71f);
	glm::vec3 sunDir = glm::vec3(1.0f, 0.3f, 0.1f);
	/// <summary>
	/// 云层高度
	/// </summary>
	float height;
	float resolutionX;
	float resolutionY;
	float time = 0;
	float speed = 0.1;// 0.1;//0.1-1;
	int seed = 1;//随机种子 默认3000种;
};


struct DeviceDescriptor
{
	vks::VulkanDevice* vulkanDevice;
	VkRenderPass renderPass;
	VkQueue queue;
	VkPipelineCache pipelineCache;
	VkSampleCountFlagBits multiSampleCount;
	VkDescriptorBufferInfo* secneBufferInfo;
	VkDescriptorBufferInfo* paramsBufferInfo;
};

enum SceneType
{
	Day = 0,
	Night = 1
};

class DynamicSky
{

public:
	SceneType Type = SceneType::Night;
	bool IsShow = true;
	CloudUBO m_paramter;
	DynamicSky(CloudParams& param, DeviceDescriptor& deviceDescriptor);
	~DynamicSky();
	void GeneateSpeed();
	//void Init(vks::VulkanDevice* device, VkQueue transferQueue);
	void Draw(VkCommandBuffer commandBuffer);
	void UpdateUniformBuffers();
#ifdef AUTOUPDATEShader
	void UpdateShader();
	void UpdateShader(std::string shader, VkPipeline& pipe);
#endif // AUTOUPDATEShader


private:
	vkglTF::Model::Vertices vertices;
	vkglTF::Model::Indices indices;

	std::string m_dataPath = "";
#ifdef AUTOUPDATEShader
	time_t fileDog = 0;
	std::map<std::string, time_t> dog_map;
#endif // AUTOUPDATEShader
	struct Buffer {
		VkDevice device;
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkDescriptorBufferInfo descriptor;
		int32_t count = 0;
		void* mapped = nullptr;
		void create(vks::VulkanDevice* device, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, bool map = true) {
			this->device = device->logicalDevice;
			device->createBuffer(usageFlags, memoryPropertyFlags, size, &buffer, &memory);
			descriptor = { buffer, 0, size };
			if (map) {
				VK_CHECK_RESULT(vkMapMemory(device->logicalDevice, memory, 0, size, 0, &mapped));
			}
		}
		void destroy() {
			if (mapped) {
				unmap();
			}
			vkDestroyBuffer(device, buffer, nullptr);
			vkFreeMemory(device, memory, nullptr);
			buffer = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;
		}
		void map() {
			VK_CHECK_RESULT(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
		}
		void unmap() {
			if (mapped) {
				vkUnmapMemory(device, memory);
				mapped = nullptr;
			}
		}
		void flush(VkDeviceSize size = VK_WHOLE_SIZE) {
			VkMappedMemoryRange mappedRange{};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = memory;
			mappedRange.size = size;
			VK_CHECK_RESULT(vkFlushMappedMemoryRanges(device, 1, &mappedRange));
		}
	} m_buffer;

	long long m_lastTime = 0;
	void Init(CloudParams& param, DeviceDescriptor& deviceDescriptor);
	void Destroy(VkDevice device);

	VkSampleCountFlagBits m_multiSampleCount = VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT;
	void InitVertices();
	void InitDescriptor();
	void InitPipeline();
	VkQueue* m_queue;
	VkDevice device;
	VkRenderPass m_renderPass;
	VkPipelineCache m_pipelineCache;
	VkPipeline m_cloudPipeline;
	VkPipeline m_nightPipeline;
	vks::VulkanDevice* pVulkanDevice;
	VkPipelineLayout m_pipelineLayout;
	VkDescriptorPool m_descriptorPool;

	VkDescriptorSetLayout m_secneDescriptorSetLayout;

	VkDescriptorBufferInfo* m_secneBufferInfo;
	VkDescriptorBufferInfo* m_paramsBufferInfo;
	VkDescriptorSet m_secneDescriptorSet;
};



#endif // __DYCLOUD__
