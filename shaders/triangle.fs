#version 450

layout(location = 0) in vec2 tex_coords;

// layout(binding = 0, set = 0) uniform sampler2D texture_image;

layout(location = 0) out vec4 frag_color;

void main() {
    // vec4 texture_color = texture(texture_image, tex_coords);
    vec4 texture_color = vec4(1.0, 0.0, 0.0, 1.0);
    frag_color = texture_color;
}
