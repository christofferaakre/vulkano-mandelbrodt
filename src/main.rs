use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassContents};
use vulkano::pipeline::graphics::input_assembly;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::sampler::Filter;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    image::{ImageAccess, ImageUsage},
    impl_vertex,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{vertex_input::BuffersDefinition, viewport::ViewportState},
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
    sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
    swapchain::PresentMode,
    sync::GpuFuture,
};
use vulkano_util::{
    context::VulkanoContext,
    renderer::DEFAULT_IMAGE_FORMAT,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}
impl_vertex!(Vertex, position, tex_coords);

const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [-1.0, -1.0],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [-1.0, 3.0],
        tex_coords: [0.0, 2.0],
    },
    Vertex {
        position: [3.0, -1.0],
        tex_coords: [2.0, 0.0],
    },
];

fn main() {
    let context = VulkanoContext::new(Default::default());
    let device = context.device();
    let mut event_loop = EventLoop::new();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let compute_queue = context.compute_queue();
    let graphics_queue = context.graphics_queue();

    let compute_shader = cs::load(device.clone()).unwrap();
    let vertex_shader = vs::load(device.clone()).unwrap();
    let fragment_shader = fs::load(device.clone()).unwrap();

    let width = 800;
    let height = 600;

    let mut windows = VulkanoWindows::default();
    let window_id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            width: width as f32,
            height: height as f32,
            title: "Mandelbrodt set".to_string(),
            present_mode: PresentMode::Fifo,
            ..Default::default()
        },
        |_| {},
    );

    let render_target_id = 0;
    let renderer = windows
        .get_primary_renderer_mut()
        .expect("Failed to create renderer");
    renderer.add_additional_image_view(
        render_target_id,
        DEFAULT_IMAGE_FORMAT,
        ImageUsage {
            sampled: true,
            storage: true,
            color_attachment: true,
            transfer_dst: true,
            ..Default::default()
        },
    );

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: renderer.swapchain_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let graphics_pipeline = GraphicsPipeline::start()
        .render_pass(subpass.clone())
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(Default::default())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .build(device.clone())
        .unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        VERTICES,
    )
    .unwrap();

    let mut recreate_swapchain = false;

    loop {
        event_loop.run_return(|event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Wait;
                }
                WindowEvent::Resized(_) => {
                    recreate_swapchain = true;
                }
                _ => {}
            },
            Event::RedrawEventsCleared => {
                let dimensions = renderer.window_size();
                let width = dimensions[0];
                let height = dimensions[1];

                if width == 0.0 || height == 0.0 {
                    return;
                }

                let acquire_future = match renderer.acquire() {
                    Ok(future) => future,
                    Err(e) => {
                        println!("{}", e);
                        return;
                    }
                };

                let image_view = renderer.get_additional_image_view(render_target_id);

                let mut compute_command_buffer_builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    graphics_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let set_layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    set_layout.clone(),
                    [WriteDescriptorSet::image_view(0, image_view.clone())],
                )
                .unwrap();

                let img_dims = image_view.image().dimensions().width_height();

                compute_command_buffer_builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0,
                        set,
                    )
                    .dispatch([img_dims[0] / 8, img_dims[1] / 8, 1])
                    .unwrap();

                let compute_command_buffer = compute_command_buffer_builder.build().unwrap();

                let compute_future = compute_command_buffer
                    .execute(graphics_queue.clone())
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .join(acquire_future);

                let sampler = Sampler::new(
                    device.clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        mipmap_mode: SamplerMipmapMode::Linear,
                        ..Default::default()
                    },
                )
                .unwrap();

                let set_layout = graphics_pipeline.layout().set_layouts().get(0).unwrap();
                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    set_layout.clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        image_view.clone(),
                        sampler.clone(),
                    )],
                )
                .unwrap();

                let mut graphics_command_buffer_builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    graphics_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let target = renderer.swapchain_image_view();

                let framebuffer = Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![target],
                        ..Default::default()
                    },
                )
                .unwrap();

                let viewport = Viewport {
                    origin: [0.0, 0.0],
                    dimensions: renderer.window_size(),
                    depth_range: 0.0..1.0,
                };

                graphics_command_buffer_builder
                    .bind_pipeline_graphics(graphics_pipeline.clone())
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            render_pass: render_pass.clone(),
                            clear_values: vec![Some([0.3, 0.3, 0.3, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        graphics_pipeline.layout().clone(),
                        0,
                        set,
                    )
                    .set_viewport(0, [viewport])
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(VERTICES.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let graphics_command_buffer = graphics_command_buffer_builder.build().unwrap();

                let after_future = compute_future
                    .then_execute(graphics_queue.clone(), graphics_command_buffer)
                    .unwrap()
                    .boxed();

                renderer.present(after_future, true);
            }
            _ => {}
        });
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 tex_coords;

        layout(location = 0) out vec2 f_tex_coords;

        void main() {
            gl_Position = vec4(position.xy, 0.0, 1.0);
            f_tex_coords = tex_coords;;
        }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450

        layout(location = 0) in vec2 tex_coords;

        layout(location = 0) out vec4 f_color;

        layout(set = 0, binding = 0) uniform sampler2D tex;

        void main() {
            f_color = texture(tex, tex_coords);
        }

        "
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        layout(binding = 0, set = 0, rgba8) uniform writeonly image2D image;

        vec2 iterate(vec2 z, vec2 c) {
            return vec2(
                z.x * z.x - z.y * z.y + c.x,
                2 * z.x * z.y + c.y
            );
        }

        void main() {
            vec2 dims = vec2(imageSize(image));
            //float ar = dims.x / dims.y;
            float x_norm = (gl_GlobalInvocationID.x / dims.x);
            float y_norm = (gl_GlobalInvocationID.y / dims.y);

            float x0 = (x_norm * 4.0) - 2.0;
            float y0 = (y_norm * 4.0) - 2.0;

            float i;

            vec2 c = vec2(x0, y0);
            vec2 z = c;
            for(i = 0.0; i < 1.0; i += 0.005) {
                z = iterate(z, c);
                if (length(z) > 8.0) {
                    break;
                }
            }

            vec4 pixel = vec4(i, i, i, 1.0);
            imageStore(image, ivec2(gl_GlobalInvocationID.xy), pixel);
        }
        "
    }
}
