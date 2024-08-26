glslangValidator.exe -V .\dynamicsky.vert
del dynamicsky.vert.spv
move vert.spv dynamicsky.vert.spv

glslangValidator.exe -V .\dynamicday.frag
del dynamicday.frag.spv
move frag.spv dynamicday.frag.spv



glslangValidator.exe -V .\dynamicnight.frag
del dynamicnight.frag.spv
move frag.spv dynamicnight.frag.spv