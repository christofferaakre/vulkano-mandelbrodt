#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D image;

void main() {
    vec4 pixel = vec4(1.0, 0.0, 0.0, 1.0);
    imageStore(image, ivec2(gl_GlobalInvocationID.xy), pixel);
}
