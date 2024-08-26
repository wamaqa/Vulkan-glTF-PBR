#ifdef STB_IMAGE_IMPLEMENTATION
	#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "DynamicSky.h"

#include <stb_image.h>
#include <chrono>

#define Radius            10.55f//0.36355339f
#define BoxMoveScele      2.2f//0.72710678f
#define PlaneMoveScele    1.1f//0.72710678f
#define MoveScele      (isBox ? BoxMoveScele :  PlaneMoveScele)
#define LengthScale  1.0f//1.41421356f


struct VertexInfo
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

static void LoadTexture(vks::Texture2D& texture, std::string& path, vks::VulkanDevice* device, VkQueue queue)
{
	int imgWidth, imgHeight, imgChannels;
	void* buff =stbi_load(path.c_str(), &imgWidth, &imgHeight, NULL, 4);
	texture.loadFromBuffer(buff, imgHeight * imgWidth * 4, VK_FORMAT_R8G8B8A8_UNORM, imgWidth, imgHeight, device, queue);
	stbi_image_free(buff);
}

DynamicSky::DynamicSky(CloudParams& param, DeviceDescriptor& deviceDescriptor)
{
	Init(param, deviceDescriptor);
}

DynamicSky::~DynamicSky()
{
	if(this)
		Destroy(device);
}

inline VkPipelineShaderStageCreateInfo loadShaders(VkDevice device, std::string filename, VkShaderStageFlagBits stage)
{
	VkPipelineShaderStageCreateInfo shaderStage{};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = stage;
	shaderStage.pName = "main";

	std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

	if (is.is_open()) {
		size_t size = is.tellg();
		is.seekg(0, std::ios::beg);
		char* shaderCode = new char[size];
		is.read(shaderCode, size);
		is.close();
		assert(size > 0);
		VkShaderModuleCreateInfo moduleCreateInfo{};
		moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleCreateInfo.codeSize = size;
		moduleCreateInfo.pCode = (uint32_t*)shaderCode;
		vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderStage.module);
		delete[] shaderCode;
	}
	else {
		std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
		shaderStage.module = VK_NULL_HANDLE;
	}

	assert(shaderStage.module != VK_NULL_HANDLE);
	return shaderStage;
}

void DynamicSky::Init(CloudParams& param, DeviceDescriptor& deviceDescriptor)
{
	pVulkanDevice = deviceDescriptor.vulkanDevice;
	m_queue = &deviceDescriptor.queue;
	this->device = deviceDescriptor.vulkanDevice->logicalDevice;
	m_multiSampleCount = deviceDescriptor.multiSampleCount;
	m_renderPass = deviceDescriptor.renderPass;
	m_pipelineCache = deviceDescriptor.pipelineCache;
	m_secneBufferInfo = deviceDescriptor.secneBufferInfo;
	m_paramsBufferInfo = deviceDescriptor.paramsBufferInfo;
	m_paramter.height = param.height;
	m_paramter.resolutionX = param.resolutionX;
	m_paramter.resolutionY = param.resolutionY;
	m_paramter.seed = rand() % 3000;

	m_paramter.sunDir = glm::vec3(1.0f, 0.6f, 0.1f);
	m_paramter.sunColor = glm::vec3(0.9, 0.9, 0.9);
	//m_paramter.skyColor = glm::vec3(21.0 / 250, 41.0 / 255, 86.0 / 255);


	m_dataPath = param.dataPath;
	InitVertices();
	InitDescriptor();
	InitPipeline();
}

void DynamicSky::GeneateSpeed()
{
	m_paramter.seed = rand() % 3000;
}
void DynamicSky::Draw(VkCommandBuffer cmdBuffer)
{
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
#ifdef AUTOUPDATEShader
	if (ms.count() - fileDog > 1000)
	{
		fileDog = ms.count();
		UpdateShader();
	}
#endif // AUTOUPDATEShader


	if (!IsShow) return;
	vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Type == SceneType::Day ? m_cloudPipeline : m_nightPipeline);

	vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_secneDescriptorSet, 0, nullptr);

	const VkDeviceSize offsets[1] = { 0 };
	vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertices.buffer, offsets);
	vkCmdBindIndexBuffer(cmdBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(cmdBuffer, 6, 1, 0, 0, 0);

}


void DynamicSky::UpdateUniformBuffers()
{
	if (this)
	{
		std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
		m_paramter.skyColor = Type == SceneType::Day ? glm::vec3(0.18f, 0.22f, 0.4f) : glm::vec3(0.05, 0.1, 0.2);
		if (ms.count() - m_lastTime > 10)
		{
			m_lastTime = ms.count();
			time_t t;
			time(&t);
			m_paramter.time += 0.00001 * m_paramter.speed * m_paramter.resolutionX;//0.001;
			if (m_paramter.time > m_paramter.resolutionX)
				m_paramter.time = 0.0;
			memcpy(m_buffer.mapped, &m_paramter, sizeof(CloudUBO));
		}
	
	}
}


#ifdef AUTOUPDATEShader
void DynamicSky::UpdateShader()
{
	UpdateShader(m_dataPath + "shaders/dynamicnight.frag", m_nightPipeline);
	UpdateShader(m_dataPath + "shaders/dynamicday.frag", m_cloudPipeline);
	vkDeviceWaitIdle(device);
}

void DynamicSky::UpdateShader(std::string shaderFile, VkPipeline& pipe)
{
		//std::string file1 = m_dataPath + "shaders/dynamicnight.frag";
	struct _stat stat_buffer;
	struct tm tmStruct;
	char timeChar[26];
	int result = _stat(shaderFile.c_str(), &stat_buffer);
	if (dog_map.find(shaderFile) == dog_map.end())
	{
		dog_map[shaderFile] = stat_buffer.st_mtime;
	}
	else if (dog_map[shaderFile] == stat_buffer.st_mtime)
	{
		return;
	}

	if (dog_map[shaderFile] == 0) dog_map[shaderFile] = stat_buffer.st_mtime;
	if (dog_map[shaderFile] != stat_buffer.st_mtime)
	{
		dog_map[shaderFile] = stat_buffer.st_mtime;
		localtime_s(&tmStruct, &stat_buffer.st_mtime);
		strftime(timeChar, sizeof(timeChar), "%Y-%m-%d %H:%M:%S", &tmStruct);
		std::cout << "修改时间: " << timeChar << std::endl;
		std::string cmd = "glslangValidator.exe -V " + shaderFile + " -o " + shaderFile + ".spv";
		system(cmd.c_str());

		std::ifstream is(shaderFile + ".spv", std::ios::binary | std::ios::in | std::ios::ate);
		if (is.is_open())
		{
			if (pipe)vkDestroyPipeline(device, pipe, nullptr);
			InitPipeline();
		}
		else
		{
			std::cerr << "Error: Could not open shader file \"" << shaderFile << "\"" << std::endl;
		}
		is.close();
		vkDeviceWaitIdle(device);
	}
}
#endif
void DynamicSky::Destroy(VkDevice device)
{
	if (vertices.buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(device, vertices.buffer, nullptr);
		vkFreeMemory(device, vertices.memory, nullptr);
		vertices.buffer = VK_NULL_HANDLE;
	}
	if (indices.buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(device, indices.buffer, nullptr);
		vkFreeMemory(device, indices.memory, nullptr);
		indices.buffer = VK_NULL_HANDLE;
	}
	if (indices.buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(device, indices.buffer, nullptr);
		vkFreeMemory(device, indices.memory, nullptr);
		indices.buffer = VK_NULL_HANDLE;
	}

	if(m_descriptorPool) vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
	if(m_secneDescriptorSetLayout) vkDestroyDescriptorSetLayout(device, m_secneDescriptorSetLayout, nullptr);
	if(m_cloudPipeline)vkDestroyPipeline(device, m_cloudPipeline, nullptr);
	if(m_nightPipeline)vkDestroyPipeline(device, m_nightPipeline, nullptr);
	if(m_pipelineLayout)vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
}

void DynamicSky::InitVertices()
{

	m_buffer.create(pVulkanDevice, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sizeof(CloudUBO));
	memcpy(m_buffer.mapped, &m_paramter, sizeof(CloudUBO));
	size_t vertexCount = 4;
	size_t indexCount = 6;

	size_t vertexBufferSize = vertexCount * sizeof(VertexInfo);
	size_t indexBufferSize = indexCount * sizeof(uint32_t);


	VertexInfo* vertexBuffer = new VertexInfo[vertexCount];


	vertexBuffer[0] = { { -1,-1, 0}, { 0, 0, 0}, {  1, 0 } }; //CreateVertex(glm::vec3(-1, -1, 0), glm::vec4(1, 1, 1, 0.2));
	vertexBuffer[1] = { { -1, 1, 0}, { 0, 0, 0}, {  0, 0 } };//CreateVertex(glm::vec3(-1, 1, 0), glm::vec4(1, 1, 1, 0.2));
	vertexBuffer[2] = { {  1, 1, 0}, { 0, 0, 0}, {  0, 1 } };//CreateVertex(glm::vec3(1, 1, 0), glm::vec4(1, 1, 1, 0.2));
	vertexBuffer[3] = { {  1,-1, 0}, { 0, 0, 0}, {  1, 1 } };//CreateVertex(glm::vec3(1, -1, 0), glm::vec4(1, 1, 1, 0.2));

	uint32_t* indexBuffer = new uint32_t[indexCount]{ 0, 1, 2, 2, 3, 0};

	struct StagingBuffer {
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertexStaging, indexStaging;


	pVulkanDevice->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		vertexBufferSize,
		&vertexStaging.buffer,
		&vertexStaging.memory,
		vertexBuffer);

	pVulkanDevice->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		indexBufferSize,
		&indexStaging.buffer,
		&indexStaging.memory,
		indexBuffer);

	pVulkanDevice->createBuffer(
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		vertexBufferSize,
		&vertices.buffer,
		&vertices.memory);
	// Index buffer
	pVulkanDevice->createBuffer(
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		indexBufferSize,
		&indices.buffer,
		&indices.memory);

	// Copy from staging buffers
	VkCommandBuffer copyCmd = pVulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	VkBufferCopy copyRegion = {};

	copyRegion.size = vertexBufferSize;
	vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

	if (indexBufferSize > 0) {
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);
	}

	pVulkanDevice->flushCommandBuffer(copyCmd, *m_queue, true);

	vkDestroyBuffer(pVulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
	vkFreeMemory(pVulkanDevice->logicalDevice, vertexStaging.memory, nullptr);
	if (indexBufferSize > 0) {
		vkDestroyBuffer(pVulkanDevice->logicalDevice, indexStaging.buffer, nullptr);
		vkFreeMemory(pVulkanDevice->logicalDevice, indexStaging.memory, nullptr);
	}

	delete[] vertexBuffer;
	delete[] indexBuffer;
}


inline VkWriteDescriptorSet CreateWriteDescriptorSet(uint32_t index, VkStructureType type, VkDescriptorType descriptorType, uint32_t descriptorCount, VkDescriptorSet& dstSet, VkDescriptorImageInfo* pImageInfo)
{
	VkWriteDescriptorSet writeDescriptorSet{};
	writeDescriptorSet.sType = type;
	writeDescriptorSet.descriptorType = descriptorType;
	writeDescriptorSet.descriptorCount = descriptorCount;
	writeDescriptorSet.dstSet = dstSet;
	writeDescriptorSet.dstBinding = index;
	writeDescriptorSet.pImageInfo = pImageInfo;
	return writeDescriptorSet;
}
void DynamicSky::InitDescriptor()
{
	/*
Descriptor pool
*/
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 50 }
	};
	VkDescriptorPoolCreateInfo descriptorPoolCI{};
	descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCI.poolSizeCount = poolSizes.size();
	descriptorPoolCI.pPoolSizes = poolSizes.data();
	descriptorPoolCI.maxSets = 20;
	VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &m_descriptorPool));

	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
					{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
					{ 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
					{ 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		};

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
		descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
		descriptorSetLayoutCI.bindingCount = setLayoutBindings.size();
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &m_secneDescriptorSetLayout));
		VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
		descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorSetAllocInfo.descriptorPool = m_descriptorPool;
		descriptorSetAllocInfo.pSetLayouts = &m_secneDescriptorSetLayout;
		descriptorSetAllocInfo.descriptorSetCount = 1;

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &m_secneDescriptorSet));

		std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};
		writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[0].descriptorCount = 1;
		writeDescriptorSets[0].dstSet = m_secneDescriptorSet;
		writeDescriptorSets[0].dstBinding = 0;
		writeDescriptorSets[0].pBufferInfo = m_secneBufferInfo;

		writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[1].descriptorCount = 1;
		writeDescriptorSets[1].dstSet = m_secneDescriptorSet;
		writeDescriptorSets[1].dstBinding = 1;
		writeDescriptorSets[1].pBufferInfo = m_paramsBufferInfo;

		writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[2].descriptorCount = 1;
		writeDescriptorSets[2].dstSet = m_secneDescriptorSet;
		writeDescriptorSets[2].dstBinding = 2;
		writeDescriptorSets[2].pBufferInfo = &m_buffer.descriptor;
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);
	}
	/*
		Pipeline layout
	*/
	//VkPushConstantRange pushConstantRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(CloudUBO) };


	VkPipelineLayoutCreateInfo pipelineLayoutCI{};
	pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCI.pushConstantRangeCount = 0;
	//pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &m_secneDescriptorSetLayout;
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &m_pipelineLayout));

}

void DynamicSky::InitPipeline()
{

	/*
		Pipeline
	*/
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
	inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
	rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationStateCI.lineWidth = 1.0f;

	VkPipelineColorBlendAttachmentState blendAttachmentState{};
	blendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAttachmentState.blendEnable = VK_TRUE;
	blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
	blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
	colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendStateCI.attachmentCount = 1;
	colorBlendStateCI.pAttachments = &blendAttachmentState;

	VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
	depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilStateCI.depthTestEnable = VK_FALSE;
	depthStencilStateCI.depthWriteEnable = VK_FALSE;
	depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencilStateCI.front = depthStencilStateCI.back;
	depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;

	VkPipelineViewportStateCreateInfo viewportStateCI{};
	viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateCI.viewportCount = 1;
	viewportStateCI.scissorCount = 1;

	VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
	multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

	if (m_multiSampleCount > VK_SAMPLE_COUNT_1_BIT) {
		multisampleStateCI.rasterizationSamples = m_multiSampleCount;
	}

	std::vector<VkDynamicState> dynamicStateEnables = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};
	VkPipelineDynamicStateCreateInfo dynamicStateCI{};
	dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
	dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

	VkVertexInputBindingDescription vertexInputBinding = { 0, sizeof(VertexInfo), VK_VERTEX_INPUT_RATE_VERTEX };
	std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexInfo, pos)},
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexInfo, normal) },
		{ 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexInfo, uv) },
	};

	VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
	vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputStateCI.vertexBindingDescriptionCount = 1;
	vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
	vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
	vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

	rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
	VkGraphicsPipelineCreateInfo pipelineCI{};
	pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCI.layout = m_pipelineLayout;
	pipelineCI.renderPass = m_renderPass;
	pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
	pipelineCI.pVertexInputState = &vertexInputStateCI;
	pipelineCI.pRasterizationState = &rasterizationStateCI;
	pipelineCI.pColorBlendState = &colorBlendStateCI;
	pipelineCI.pMultisampleState = &multisampleStateCI;
	pipelineCI.pViewportState = &viewportStateCI;
	pipelineCI.pDepthStencilState = &depthStencilStateCI;
	pipelineCI.pDynamicState = &dynamicStateCI;
	pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineCI.pStages = shaderStages.data();

	shaderStages = {
		loadShaders(device, m_dataPath + "shaders/dynamicsky.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
		loadShaders(device, m_dataPath + "shaders/dynamicday.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
	};
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineCI, nullptr, &m_cloudPipeline));


	shaderStages = {
		loadShaders(device, m_dataPath + "shaders/dynamicsky.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
		loadShaders(device, m_dataPath + "shaders/dynamicnight.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
	};
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineCI, nullptr, &m_nightPipeline));

	for (auto shaderStage : shaderStages) {
		vkDestroyShaderModule(device, shaderStage.module, nullptr);
	}
}

