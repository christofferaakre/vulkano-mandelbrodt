use std::sync::Arc;

use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, RenderPassBeginInfo,
        SecondaryAutoCommandBuffer,
    },
    device::{Device, Queue},
    impl_vertex,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{vertex_input::BuffersDefinition, viewport::ViewportState},
        GraphicsPipeline,
    },
    render_pass::Subpass,
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Default, Zeroable, Pod)]
pub struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}
impl_vertex!(Vertex, position, tex_coords);

const VERTICES: [Vertex; 4] = [
    Vertex {
        position: [-0.5, -0.5],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5],
        tex_coords: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5],
        tex_coords: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5],
        tex_coords: [1.0, 1.0],
    },
];

pub struct DrawPipeline {
    memory_allocator: StandardMemoryAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
}

impl DrawPipeline {
    pub fn new(device: &Arc<Device>, queue: Arc<Queue>, subpass: Subpass) -> Self {
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();

        let pipeline = GraphicsPipeline::start()
            .input_assembly_state(Default::default())
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .render_pass(subpass.clone())
            .build(device.clone())
            .unwrap();

        Self {
            memory_allocator,
            command_buffer_allocator,
            queue,
            subpass,
            pipeline,
        }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();

        let viewport = vulkano::pipeline::graphics::viewport::Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            vulkano::buffer::BufferUsage {
                vertex_buffer: true,
                ..Default::default()
            },
            false,
            VERTICES,
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, [viewport])
            .bind_vertex_buffers(0, vertex_buffer)
            .draw(VERTICES.len() as u32, 1, 0, 0)
            .unwrap();

        let command_buffer = builder.build().unwrap();
        command_buffer
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 tex_coords;

        layout(location = 0) out vec2 tex_coords_out;

        void main() {
            gl_Position = vec4(position.xy, 0.0, 1.0);
            tex_coords_out = tex_coords;
        }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450

        layout(location = 0) in vec2 tex_coords_v;

        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
        "
    }
}
