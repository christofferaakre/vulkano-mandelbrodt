#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D image;

vec2 iterate(vec2 z, vec2 z0) {
    return vec2(
            z.x * z.x - z.y * z.y + z0.x,
            2 * z.x * z.y + z0.y
            );
}

void main() {
    vec2 norm_coords = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(image));
    vec2 z0 = (norm_coords - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    float i;
    vec2 z = z0;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = iterate(z, z0);
        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 pixel = vec4(i, i, i, 1.0);
    imageStore(image, ivec2(gl_GlobalInvocationID.xy), pixel);

}
