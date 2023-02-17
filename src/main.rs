use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::device::DeviceCreateInfo;
use vulkano::sync::GpuFuture;
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
use winit::{event_loop::EventLoop, window::WindowBuilder};

fn select_physical_device(
    instance: &Arc<Instance>,
    // surface: &Arc<Surface>,
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
                    q.queue_flags.compute
                    // q.queue_flags.compute && p.surface_support(i as u32, &surface).unwrap_or(false)
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

    // let surface = WindowBuilder::new()
    //     .build_vk_surface(&event_loop, instance.clone())
    //     .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..Default::default()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &device_extensions);

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

    let buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        vulkano::buffer::BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..(image.mem_size() as u32)).map(|_| 0u8),
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

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        OneTimeSubmit,
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

    builder
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

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_contents = buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, buffer_contents).unwrap();

    let save_path = "out/mandelbrodt.jpg";
    image.save(save_path).unwrap();

    println!("saved image to {}", save_path);
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/mandelbrodt.cs"
    }
}
