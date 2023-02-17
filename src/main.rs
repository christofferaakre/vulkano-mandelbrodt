use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Rgba};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
};
use vulkano::device::{DeviceCreateInfo, Queue};
use vulkano::image::{ImmutableImage, SwapchainImage};
use vulkano::impl_vertex;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture};
use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
        AutoCommandBufferBuilder,
        CommandBufferUsage::OneTimeSubmit,
    },
    descriptor_set::{
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, StorageImage},
    instance::Instance,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        ComputePipeline, Pipeline,
        PipelineBindPoint::{self, Compute},
    },
    swapchain::Surface,
    sync, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;
use winit::{event_loop::EventLoop, window::WindowBuilder};

#[repr(C)]
#[derive(Copy, Clone, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

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

impl_vertex! {Vertex, position, tex_coords}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.compute && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 3,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("No devices available")
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}

fn get_graphics_command_buffers(
    device: Arc<Device>,
    allocator: &StandardCommandBufferAllocator,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
    viewport: &Viewport,
    image_views: &Vec<Arc<ImageView<ImmutableImage>>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .enumerate()
        .map(|(i, framebuffer)| {
            let set_layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                descriptor_set_allocator,
                set_layout.clone(),
                [WriteDescriptorSet::image_view(0, image_views[i].clone())],
            )
            .unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.3, 0.3, 0.3, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                // .bind_descriptor_sets(
                //     PipelineBindPoint::Graphics,
                //     pipeline.layout().clone(),
                //     0,
                //     set,
                // )
                .set_viewport(0, [viewport.clone()])
                .draw(3, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        vulkano::instance::InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();

    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..Default::default()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    println!(
        "Using physical device: {}, type: {:?}",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let compute_shader = cs::load(device.clone()).unwrap();
    let vertex_shader = vs::load(device.clone()).unwrap();
    let fragment_shader = fs::load(device.clone()).unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let image = StorageImage::new(
        &memory_allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        [queue_family_index],
    )
    .expect("Failed to create image");

    let view = ImageView::new_default(image.clone()).unwrap();

    let buffer_size = image.mem_size() as u32;

    let buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        vulkano::buffer::BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..buffer_size).map(|_| 0u8),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let mut compute_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    let set_layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        set_layout.clone(),
        vec![WriteDescriptorSet::image_view(
            queue_family_index,
            view.clone(),
        )],
    )
    .unwrap();

    compute_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(
            vulkano::command_buffer::CopyImageToBufferInfo::image_buffer(
                image.clone(),
                buffer.clone(),
            ),
        )
        .unwrap();

    let command_buffer = compute_builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_contents = buffer.read().unwrap();

    // graphics stuff
    let surface_capabilites = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let image_format = Some(
        physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        vulkano::swapchain::SwapchainCreateInfo {
            min_image_count: surface_capabilites.min_image_count + 1,
            image_format,
            image_extent: window.inner_size().into(),
            composite_alpha: surface_capabilites
                .supported_composite_alpha
                .iter()
                .next()
                .unwrap(),
            image_usage: vulkano::image::ImageUsage {
                color_attachment: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
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

    let buf_contents = &buffer_contents[..];

    let mut image_buffers = vec![];
    for _ in 0..images.len() {
        let image_buffer = ImmutableImage::from_iter(
            &memory_allocator,
            (0..buffer_size).map(|i| buffer_contents[i as usize]),
            vulkano::image::ImageDimensions::Dim2d {
                height: 1024,
                width: 1024,
                array_layers: 1,
            },
            vulkano::image::MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            &mut AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap(),
        )
        .unwrap();

        image_buffers.push(image_buffer);
    }

    let views = image_buffers
        .iter()
        .map(|image_buffer| {
            let view = ImageView::new_default(image_buffer.clone()).unwrap();
            view
        })
        .collect::<Vec<_>>();

    let render_pass = get_render_pass(device.clone(), &swapchain);

    let graphics_pipeline = GraphicsPipeline::start()
        .input_assembly_state(Default::default())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut framebuffers = get_framebuffers(&images, &render_pass);

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut graphics_command_buffers = get_graphics_command_buffers(
        device.clone(),
        &command_buffer_allocator,
        &descriptor_set_allocator,
        &queue,
        &graphics_pipeline,
        &framebuffers,
        &vertex_buffer,
        &viewport,
        &views,
    );

    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,

        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }

        Event::MainEventsCleared => {}

        Event::RedrawEventsCleared => {
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let dimensions = window.inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            let previous_fence = &mut fences[previous_fence_i];
            if let Some(fence) = previous_fence.as_mut() {
                fence.cleanup_finished();
            }

            if recreate_swapchain {
                recreate_swapchain = false;

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain {:?}", e),
                };
                swapchain = new_swapchain;

                framebuffers = get_framebuffers(&new_images, &render_pass);

                viewport.dimensions = dimensions.into();
            }

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            graphics_command_buffers = get_graphics_command_buffers(
                device.clone(),
                &command_buffer_allocator,
                &descriptor_set_allocator,
                &queue,
                &graphics_pipeline,
                &framebuffers,
                &vertex_buffer,
                &viewport,
                &views,
            );

            // wait for the fence related to this image to finish
            // normally this would the be the oldest fence, that most like has already
            // finished
            if let Some(image_fence) = &fences[image_index as usize] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();
                    now.boxed()
                }
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(
                    queue.clone(),
                    graphics_command_buffers[image_index as usize].clone(),
                )
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo {
                        image_index,
                        // ..SwapchainPresentInfo::swapchain(swapchain.clone())
                        ..SwapchainPresentInfo::swapchain_image_index(
                            swapchain.clone(),
                            image_index,
                        )
                    },
                )
                .then_signal_fence_and_flush();

            fences[image_index as usize] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("Failed to flush future {:?}", e);
                    None
                }
            };

            previous_fence_i = image_index as usize;
        }

        _ => {}
    })
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/mandelbrodt.cs"
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/triangle.vs"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/triangle.fs"
    }
}
