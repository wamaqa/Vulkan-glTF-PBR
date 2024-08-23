glslangValidator.exe -V .\dynamiccloud.frag
del dynamiccloud.frag.spv
move frag.spv dynamiccloud.frag.spv

glslangValidator.exe -V .\dynamiccloud.vert
del dynamiccloud.vert.spv
move vert.spv dynamiccloud.vert.spv

glslangValidator.exe -V .\dynamicnight.frag
del dynamicnight.frag.spv
move frag.spv dynamicnight.frag.spv