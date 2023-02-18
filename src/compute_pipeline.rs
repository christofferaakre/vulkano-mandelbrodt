use std::sync::Arc;

use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::ImageAccess;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    device::{Device, Queue},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};
use vulkano_util::renderer::DeviceImageView;

pub struct MandelbrodtComputePipeline {
    pipeline: Arc<ComputePipeline>,
    queue: Arc<Queue>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
}

impl MandelbrodtComputePipeline {
    pub fn new(device: &Arc<Device>, queue: Arc<Queue>) -> Self {
        let compute_shader = cs::load(device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            device.clone(),
            compute_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap();

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

        MandelbrodtComputePipeline {
            pipeline,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn compute(&self, image: DeviceImageView) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let dimensions = image.image().dimensions().width_height();

        let desc_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [WriteDescriptorSet::image_view(0, image.clone())],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch([dimensions[0] / 8, dimensions[1] / 8, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450

        layout(binding = 0, set = 0, rgba8) uniform writeonly image2D img;

        void main() {
            vec4 pixel = vec4(1.0, 0.0, 0.0, 1.0);
            imageStore(img, ivec2(gl_GlobalInvocationID.xy), pixel);
        }
        "
    }
}
