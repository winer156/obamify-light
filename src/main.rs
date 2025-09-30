//#![windows_subsystem = "windows"]

mod morph_sim;
use std::{
    fs::File,
    num::NonZeroU64,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32},
        mpsc, Arc, RwLock,
    },
};

use bytemuck::{Pod, Zeroable};
use color_quant::NeuQuant;
use eframe::{egui, App, CreationContext, Frame, NativeOptions};
use egui::{frame, Color32, Modal, ViewportBuilder, Window};
use egui_wgpu::{self, wgpu};
use image::buffer::ConvertBuffer;
use wgpu::util::DeviceExt;

use crate::{
    calculate::{drawing_process::PixelData, GenerationSettings, ProgressMsg},
    morph_sim::{preset_path_to_name, Sim},
};

const WG_SIZE_XY: u32 = 8;
const WG_SIZE_SEEDS: u32 = 256;
//const INVALID_ID: u32 = 0xFFFF_FFFF;

mod calculate;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SeedPos {
    xy: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SeedColor {
    rgba: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsCommon {
    width: u32,
    height: u32,
    n_seeds: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsJfa {
    width: u32,
    height: u32,
    step: u32,
    _pad: u32,
}

const DEFAULT_RESOLUTION: u32 = 2048;
const GIF_FRAMERATE: u32 = 8;
const GIF_RESOLUTION: u32 = 460;
const GIF_NUM_FRAMES: u32 = 140;
const GIF_SPEED: f32 = 1.5;
const GIF_PALETTE_SAMPLEFAC: i32 = 1;

#[derive(Clone, Debug)]
enum GifStatus {
    None,
    Recording(Option<PathBuf>),
    Complete(PathBuf),
    Error(String),
}
impl GifStatus {
    fn is_recording(&self) -> bool {
        matches!(self, GifStatus::Recording(_))
    }

    fn not_recording(&self) -> bool {
        matches!(self, GifStatus::None)
    }
}

pub enum GuiMode {
    Transform,
    Draw,
}

pub struct VoronoiApp {
    prev_frame_time: std::time::Instant,
    // UI state
    size: (u32, u32),
    seed_count: u32,

    progress_tx: mpsc::SyncSender<ProgressMsg>,
    progress_rx: mpsc::Receiver<ProgressMsg>,

    gif_status: GifStatus,
    gif_encoder: Option<gif::Encoder<File>>,
    gif_palette: Option<NeuQuant>,
    gif_frame_count: u32,
    sim: Sim,

    // Seeds CPU copy
    seeds: Vec<SeedPos>,
    colors: Arc<RwLock<Vec<SeedColor>>>,
    pixeldata: Arc<RwLock<Vec<PixelData>>>,

    // EGUI texture id for presenting the shaded RGBA texture
    egui_tex_id: Option<egui::TextureId>,

    // GPU resources (lifetime tied to eframe's RenderState device)
    // Buffers
    seed_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    params_common_buf: wgpu::Buffer,
    params_jfa_buf: wgpu::Buffer,

    // Textures & views
    ids_a: wgpu::Texture,
    ids_b: wgpu::Texture,
    ids_a_view: wgpu::TextureView,
    ids_b_view: wgpu::TextureView,

    // Color (linear storage + srgb view for egui)
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,

    // Pipelines
    clear_pipeline: wgpu::ComputePipeline,
    seed_splat_pipeline: wgpu::ComputePipeline,
    jfa_pipeline: wgpu::ComputePipeline,
    shade_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    clear_bgl: wgpu::BindGroupLayout,
    seed_bgl: wgpu::BindGroupLayout,
    jfa_bgl: wgpu::BindGroupLayout,
    shade_bgl: wgpu::BindGroupLayout,

    // Bind groups that are re-created when textures change
    clear_bg_a: wgpu::BindGroup,
    clear_bg_b: wgpu::BindGroup,
    seed_bg: wgpu::BindGroup,
    jfa_bg_a_to_b: wgpu::BindGroup,
    jfa_bg_b_to_a: wgpu::BindGroup,
    shade_bg: wgpu::BindGroup,
    preview_image: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>,

    stroke_count: u32,

    frame_count: u32,

    gui: GuiState,
    current_drawing_id: Arc<AtomicU32>,
}

impl VoronoiApp {
    fn apply_sim_init(
        &mut self,
        device: &wgpu::Device,
        seed_count: u32,
        seeds: Vec<SeedPos>,
        colors: Vec<SeedColor>,
        sim: Sim,
    ) {
        self.seed_count = seed_count;
        self.seeds = seeds;
        self.sim = sim;

        // Update GPU buffers
        self.seed_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seeds"),
            contents: bytemuck::cast_slice(&self.seeds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        *self.colors.write().unwrap() = colors;
        *self.pixeldata.write().unwrap() = PixelData::init_canvas(self.frame_count);

        self.rebuild_bind_groups(device);
    }

    pub fn change_sim(&mut self, device: &wgpu::Device, source: PathBuf) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_image(self.size.0, source);
        self.apply_sim_init(device, seed_count, seeds, colors, sim);
    }

    pub fn canvas_sim(&mut self, device: &wgpu::Device, source: PathBuf) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_canvas(self.size.0, source);
        self.apply_sim_init(device, seed_count, seeds, colors, sim);
    }

    pub fn new(cc: &CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be built with the 'wgpu' feature and Renderer::Wgpu")
            .clone();
        let device = &rs.device;
        let size = (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION);
        // let seed_count = 4096;
        // let mut seeds = Vec::with_capacity(seed_count as usize);
        // let mut colors = Vec::with_capacity(seed_count as usize);
        // for x in 0..64 {
        //     for y in 0..64 {
        //         seeds.push(SeedPos {
        //             xy: [
        //                 (x as f32 + 0.5 + rng.gen::<f32>()) * (size.0 as f32 / 64.0),
        //                 (y as f32 + 0.5 + rng.gen::<f32>()) * (size.1 as f32 / 64.0),
        //             ],
        //             // rgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][rng.gen_range(0..3)],
        //             // _pad: [0.0, 0.0, 0.0],
        //         });

        //         colors.push(SeedColor {
        //             rgba: [
        //                 [1.0, 0.0, 0.0, 1.0],
        //                 [0.0, 1.0, 0.0, 1.0],
        //                 [0.0, 0.0, 1.0, 1.0],
        //             ][rng.gen_range(0..3)],
        //         });
        //     }
        // }
        // seeds.push(SeedPos {
        //     xy: [size.1 as f32 * 0.55, size.1 as f32 * 0.55],
        // });
        // seeds.push(SeedPos {
        //     xy: [size.1 as f32 * 0.45, size.1 as f32 * 0.45],
        // });

        // get all folders in ../presets
        let presets: Vec<PathBuf> = get_presets();

        let (seed_count, seeds, colors, sim) = morph_sim::init_image(
            size.0,
            presets[rand::random::<usize>() % presets.len()].clone(),
        );

        // === Buffers ===
        let seed_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seeds"),
            contents: bytemuck::cast_slice(&seeds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_common = ParamsCommon {
            width: size.0,
            height: size.1,
            n_seeds: seed_count,
            _pad: 0,
        };
        let params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: size.0,
            height: size.1,
            step: 1,
            _pad: 0,
        };
        let params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Textures ===
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, size, Some("color"));

        // === Pipelines ===
        let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_clear"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let seed_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_seed_splat"),
            entries: &[
                // seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params common
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ParamsCommon>() as u64
                        ),
                    },
                    count: None,
                },
                // dst ids
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let jfa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_jfa"),
            entries: &[
                // seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // src ids (read-only sampled as integer texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // dst ids (write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // params_jfa
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<ParamsJfa>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_shade"),
            entries: &[
                // ids texture (u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // out color (rgba8unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // seeds (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // colors
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Shader modules
        let clear_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear.wgsl").into()),
        });
        let seed_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seed_splat.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/seed.wgsl").into()),
        });
        let jfa_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jfa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jfa.wgsl").into()),
        });
        let shade_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shade.wgsl").into()),
        });

        // Pipelines
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_clear"),
                    bind_group_layouts: &[&clear_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &clear_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let seed_splat_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("seed_splat_pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("pl_seed"),
                        bind_group_layouts: &[&seed_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &seed_sm,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let jfa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jfa_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_jfa"),
                    bind_group_layouts: &[&jfa_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &jfa_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let shade_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_shade"),
                    bind_group_layouts: &[&shade_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shade_sm,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Bind groups
        let clear_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_a"),
            layout: &clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ids_a_view),
            }],
        });
        let clear_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_b"),
            layout: &clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ids_b_view),
            }],
        });

        let seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_common_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
            ],
        });

        let jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view), // will point to the final ids
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: color_buf.as_entire_binding(),
                },
            ],
        });

        let (progress_tx, progress_rx) = mpsc::sync_channel::<ProgressMsg>(1);

        Self {
            size,
            seed_count,

            seeds,
            colors: Arc::new(RwLock::new(colors)),
            pixeldata: Arc::new(RwLock::new(PixelData::init_canvas(0))),
            egui_tex_id: None,
            seed_buf,
            color_buf,
            sim,
            params_common_buf,
            params_jfa_buf,
            ids_a,
            ids_b,
            ids_a_view,
            ids_b_view,
            color_tex,
            color_view,
            clear_pipeline,
            seed_splat_pipeline,
            jfa_pipeline,
            shade_pipeline,
            clear_bgl,
            seed_bgl,
            jfa_bgl,
            shade_bgl,
            clear_bg_a,
            clear_bg_b,
            seed_bg,
            jfa_bg_a_to_b,
            jfa_bg_b_to_a,
            shade_bg,
            prev_frame_time: std::time::Instant::now(),

            progress_tx,
            progress_rx,

            gif_encoder: None,
            gif_status: GifStatus::None,
            gif_frame_count: 0,
            gif_palette: None,

            preview_image: None,

            stroke_count: 0,
            gui: GuiState::default(presets),
            frame_count: 0,
            current_drawing_id: Arc::new(AtomicU32::new(0)),
        }
    }

    fn make_ids_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ids_view"),
            format: Some(wgpu::TextureFormat::R32Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            ..Default::default()
        });
        (tex, view)
    }

    fn make_color_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn ensure_registered_texture(&mut self, rs: &egui_wgpu::RenderState) {
        if self.egui_tex_id.is_none() {
            let id = rs.renderer.write().register_native_texture(
                &rs.device,
                &self.color_view,
                wgpu::FilterMode::Linear,
            );
            self.egui_tex_id = Some(id);
        }
    }

    fn rebuild_bind_groups(&mut self, device: &wgpu::Device) {
        // Rebuild any BGs that reference texture views
        self.clear_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_a"),
            layout: &self.clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
            }],
        });
        self.clear_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clear_b"),
            layout: &self.clear_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
            }],
        });
        self.seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &self.seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_common_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
            ],
        });
        self.jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &self.shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.seed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.color_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn resize_textures(&mut self, device: &wgpu::Device, new_size: (u32, u32), rebuild_bg: bool) {
        self.size = new_size;
        // Recreate textures
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, self.size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, self.size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, self.size, Some("color"));
        self.ids_a = ids_a;
        self.ids_a_view = ids_a_view;
        self.ids_b = ids_b;
        self.ids_b_view = ids_b_view;
        self.color_tex = color_tex;
        self.color_view = color_view;

        // Update params_common
        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: self.size.0,
            height: self.size.1,
            step: 1,
            _pad: 0,
        };

        self.params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        if rebuild_bg {
            self.rebuild_bind_groups(device);
        }

        // Force re-registering the egui texture
        self.egui_tex_id = None;
    }

    fn run_gpu(&mut self, rs: &egui_wgpu::RenderState) {
        let device = &rs.device;

        // Prepare commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voronoi_jfa_encoder"),
        });

        // 1) Clear both ID textures
        for bg in [&self.clear_bg_a, &self.clear_bg_b] {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_ids_a"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            cpass.dispatch_workgroups(
                self.size.0.div_ceil(WG_SIZE_XY),
                self.size.1.div_ceil(WG_SIZE_XY),
                1,
            );
        }

        // 2) Seed splat into A
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("seed_splat"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.seed_splat_pipeline);
            cpass.set_bind_group(0, &self.seed_bg, &[]);
            cpass.dispatch_workgroups(self.seed_count.div_ceil(WG_SIZE_SEEDS), 1, 1);
            //
        }

        // 3) JFA passes, ping-pong A<->B

        let max_dim = self.size.0.max(self.size.1);
        let mut step = 1u32;
        while step < max_dim {
            step <<= 1;
        }
        step >>= 1;

        let groups_x = self.size.0.div_ceil(WG_SIZE_XY);
        let groups_y = self.size.1.div_ceil(WG_SIZE_XY);

        let mut flip = false;
        while step >= 1 {
            let pj = ParamsJfa {
                width: self.size.0,
                height: self.size.1,
                step,
                _pad: 0,
            };
            let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_jfa_staging"),
                contents: bytemuck::bytes_of(&pj),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &staging,
                0,
                &self.params_jfa_buf,
                0,
                std::mem::size_of::<ParamsJfa>() as u64,
            );
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jfa_step"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.jfa_pipeline);
                cpass.set_bind_group(
                    0,
                    if !flip {
                        &self.jfa_bg_a_to_b
                    } else {
                        &self.jfa_bg_b_to_a
                    },
                    &[],
                );
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            flip = !flip;
            step >>= 1;
        }

        // if self.refined {
        //     for _ in 0..2 {
        //         let pj = ParamsJfa {
        //             width: self.size.0,
        //             height: self.size.1,
        //             step: 1,
        //             _pad: 0,
        //         };
        //         let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("params_jfa_staging"),
        //             contents: bytemuck::bytes_of(&pj),
        //             usage: wgpu::BufferUsages::COPY_SRC,
        //         });
        //         encoder.copy_buffer_to_buffer(
        //             &staging,
        //             0,
        //             &self.params_jfa_buf,
        //             0,
        //             std::mem::size_of::<ParamsJfa>() as u64,
        //         );
        //         {
        //             let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //                 label: Some("jfa_step"),
        //                 timestamp_writes: None,
        //             });
        //             cpass.set_pipeline(&self.jfa_pipeline);
        //             cpass.set_bind_group(
        //                 0,
        //                 if !flip {
        //                     &self.jfa_bg_a_to_b
        //                 } else {
        //                     &self.jfa_bg_b_to_a
        //                 },
        //                 &[],
        //             );
        //             cpass.dispatch_workgroups(groups_x, groups_y, 1);
        //         }
        //         flip = !flip;
        //     }
        // }

        // 4) Shade to color (the final IDs are in A if flip is true, else in B).
        // Our shade BG was built with ids_a_view at binding 0. If the last write ended in B,
        // we temporarily rebind with B for this dispatch.
        let shade_with_b = flip; // if true, IDs live in B
        if shade_with_b {
            let tmp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_shade_tmp_b"),
                layout: &self.shade_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.seed_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.color_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_pipeline);
            cpass.set_bind_group(0, &tmp_bg, &[]);
            cpass.dispatch_workgroups(groups_x, groups_y, 1);
        } else {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_pipeline);
            cpass.set_bind_group(0, &self.shade_bg, &[]);
            cpass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // Submit
        rs.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn get_color_image_data(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let width = self.size.0;
        let height = self.size.1;
        let bpp = 4u32; // RGBA8
        let unpadded_bytes_per_row = width * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row as u64 * height as u64;

        // Staging buffer to receive the texture
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("color readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy color_tex -> buffer"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        slice.map_async(wgpu::MapMode::Read, move |res| {
            // res: Result<(), wgpu::BufferAsyncError>
            let _ = tx.send(res);
        });

        // Ensure the callback runs
        device.poll(wgpu::PollType::Wait)?;

        // Wait for the result and propagate any map error
        pollster::block_on(rx.receive()).expect("map_async sender dropped")?;
        let mapped = slice.get_mapped_range();
        // Remove row padding
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height as usize {
            let start = y * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            rgba.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        readback.unmap();
        Ok(rgba)
    }

    fn write_frame_to_gif(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rgba = self.get_color_image_data(device, queue)?;
        // let image =
        //     image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(self.size.0, self.size.1, rgba)
        //         .unwrap();
        // // resize to GIF_RESOLUTION
        // let resized = image::imageops::resize(
        //     &image,
        //     GIF_RESOLUTION,
        //     GIF_RESOLUTION,
        //     image::imageops::FilterType::Lanczos3,
        // );

        // if self.gif_status.is_recording() && self.gif_encoder.is_none() {
        //     resized.save("testframe.png")?;
        // }

        // let rgba = resized.into_raw();

        // if first frame, create encoder
        if self.gif_status.is_recording() && self.gif_encoder.is_none() {
            let file = rfd::FileDialog::new()
                .set_title("save gif")
                .add_filter("gif", &["gif"])
                .set_file_name(format!("{}.gif", self.sim.name()))
                .save_file();
            if let Some(path) = file {
                let colors = self
                    .colors
                    .read()
                    .unwrap()
                    .iter()
                    .flat_map(|s| s.rgba.map(|f| (f * 256.0) as u8))
                    .collect::<Vec<u8>>();
                let gif_palette = NeuQuant::new(GIF_PALETTE_SAMPLEFAC, 256, &colors);

                let file = std::fs::File::create(&path)?;
                // save a test image of the palette
                // {
                //     let mut img = image::ImageBuffer::new(16, 16);
                //     for (i, pixel) in gif_palette.color_map_rgb().chunks_exact(3).enumerate() {
                //         img.put_pixel(
                //             i as u32 % 16,
                //             i as u32 / 16,
                //             image::Rgba([pixel[0], pixel[1], pixel[2], 255]),
                //         );
                //     }
                //     img.save("palette.png")?;
                // }
                // clear file
                file.set_len(0)?;
                let mut encoder = gif::Encoder::new(
                    file,
                    GIF_RESOLUTION as u16,
                    GIF_RESOLUTION as u16,
                    &gif_palette.color_map_rgb(),
                )?;
                self.gif_palette = Some(gif_palette);
                encoder.set_repeat(gif::Repeat::Infinite)?;
                self.gif_encoder = Some(encoder);
                self.gif_frame_count = 0;
                self.gif_status = GifStatus::Recording(Some(path));
            } else {
                // cancelled
                self.stop_recording_gif(device);
                return Ok(());
            }
        }

        if let Some(encoder) = &mut self.gif_encoder {
            let nq = self.gif_palette.as_ref().unwrap();
            let pixels: Vec<u8> = rgba
                .chunks_exact(4)
                .map(|pix| nq.index_of(pix) as u8)
                .collect();
            let mut frame = gif::Frame::from_indexed_pixels(
                GIF_RESOLUTION as u16,
                GIF_RESOLUTION as u16,
                pixels,
                None,
            );

            frame.delay = ((100.0 / GIF_FRAMERATE as f32) / GIF_SPEED) as u16; // delay in 1/100 sec
            encoder.write_frame(&frame)?;
        }

        Ok(())
    }

    fn stop_recording_gif(&mut self, device: &wgpu::Device) {
        self.gif_status = GifStatus::None;
        self.gif_encoder = None;
        self.gui.animate = false;
        self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION), false);
        self.change_sim(device, self.sim.source_path());
    }

    fn draw(
        &mut self,
        last_mouse_pos: Option<(f32, f32)>,
        mousepos: (f32, f32),
        device: &wgpu::Device,
    ) {
        let stroke_id = if last_mouse_pos.is_some() {
            self.stroke_count
        } else {
            self.stroke_count += 1;
            self.stroke_count
        };
        for (i, seedpos) in self.seeds.iter().enumerate() {
            let sx = seedpos.xy[0];
            let sy = seedpos.xy[1];

            let last_mouse_pos = if let Some(a) = last_mouse_pos {
                a
            } else {
                mousepos
            };

            let dist = point_to_line_dist(
                sx,
                sy,
                last_mouse_pos.0,
                last_mouse_pos.1,
                mousepos.0,
                mousepos.1,
            );
            let thickness = if self.gui.drawing_color == [0.0, 0.0, 0.0, DRAWING_ALPHA] {
                30.0
            } else {
                50.0
            };
            let transition = 10.0;
            if dist < thickness + transition {
                let color = self.gui.drawing_color;
                let alpha =
                    ((thickness + transition - dist) / transition).clamp(0.0, 1.0) * color[3];
                let blend = |c1: f32, c2: f32, a: f32| (1.0 - a) * c1 + a * c2;
                let mut colors = self.colors.write().unwrap();
                (*colors)[i].rgba[0] = blend((*colors)[i].rgba[0], color[0], alpha);
                (*colors)[i].rgba[1] = blend((*colors)[i].rgba[1], color[1], alpha);
                (*colors)[i].rgba[2] = blend((*colors)[i].rgba[2], color[2], alpha);

                self.sim.cells[i].set_age(0);
                self.sim.cells[i].set_dst_force(0.05 + (stroke_id as f32 * 0.004).sqrt());
                self.sim.cells[i].set_stroke_id(stroke_id);
                self.pixeldata.write().unwrap()[i] = PixelData {
                    stroke_id,
                    last_edited: self.frame_count,
                };

                //self.colors[i].rgba = [0.0, 0.0, 0.0, 1.0];
            }
        }

        self.color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&self.colors.read().unwrap()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
    }

    fn handle_drawing(
        &mut self,
        ctx: &egui::Context,
        device: &wgpu::Device,
        ui: &mut egui::Ui,
        aspect: f32,
    ) {
        // get mouse position over image
        if let Some(pos) = ui.ctx().pointer_interact_pos() {
            let rect = ui.min_rect();

            if rect.contains(pos) {
                let min_y = rect.min.y;
                let min_x = rect.min.x - (rect.height() * aspect - rect.width()) / 2.0;

                let uv = (pos - egui::pos2(min_x, min_y)) / rect.height();
                let img_x = uv.x * self.size.0 as f32;
                let img_y = uv.y * self.size.1 as f32;

                if img_x > 0.0
                    && img_y > 0.0
                    && img_x < self.size.0 as f32
                    && img_y < self.size.1 as f32
                    && ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
                {
                    self.draw(self.gui.last_mouse_pos, (img_x, img_y), device);
                    self.gui.last_mouse_pos = Some((img_x, img_y));
                } else {
                    self.gui.last_mouse_pos = None;
                }
            } else {
                self.gui.last_mouse_pos = None;
            }
        } else {
            self.gui.last_mouse_pos = None;
        }
    }

    fn init_canvas(&mut self, device: &wgpu::Device) {
        let path = std::path::PathBuf::from("./blank.png");

        let settings = GenerationSettings::default();
        self.canvas_sim(device, path.clone());
        self.gui.animate = true;

        self.current_drawing_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        std::thread::spawn({
            let tx = self.progress_tx.clone();
            let colors = Arc::clone(&self.colors);
            let pixel_data = Arc::clone(&self.pixeldata);
            let frame_count = self.frame_count;
            let current_id = self.current_drawing_id.clone();
            let my_id = current_id.load(std::sync::atomic::Ordering::SeqCst);
            move || {
                let result = calculate::drawing_process::drawing_process_genetic(
                    path,
                    settings,
                    tx.clone(),
                    colors,
                    pixel_data,
                    frame_count,
                    my_id,
                    current_id,
                );
                match result {
                    Ok(()) => {}
                    Err(err) => {
                        tx.send(ProgressMsg::Error(err.to_string())).ok();
                    }
                }
            }
        });
    }
}

const DRAWING_ALPHA: f32 = 0.5;
fn point_to_line_dist(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    if dx == 0.0 && dy == 0.0 {
        // It's a point not a line segment.
        ((px - x0).powi(2) + (py - y0).powi(2)).sqrt()
    } else {
        // Calculate the t that minimizes the distance.
        let t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy);
        if t < 0.0 {
            // Beyond the 'x0,y0' end of the segment
            ((px - x0).powi(2) + (py - y0).powi(2)).sqrt()
        } else if t > 1.0 {
            // Beyond the 'x1,y1' end of the segment
            ((px - x1).powi(2) + (py - y1).powi(2)).sqrt()
        } else {
            // Projection falls on the segment
            let proj_x = x0 + t * dx;
            let proj_y = y0 + t * dy;
            ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
        }
    }
}

fn get_presets() -> Vec<PathBuf> {
    if let Ok(dir) = std::fs::read_dir("./presets") {
        dir.filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.is_dir() {
                Some(path)
            } else {
                None
            }
        })
        .collect()
    } else {
        Vec::new()
    }
}

struct GuiState {
    last_mouse_pos: Option<(f32, f32)>,
    drawing_color: [f32; 4],
    mode: GuiMode,
    animate: bool,
    fps_text: String,
    show_progress_modal: bool,
    last_progress: f32,
    process_cancelled: Arc<AtomicBool>,
    quick_process: bool,
    currently_processing: Option<PathBuf>,
    presets: Vec<PathBuf>,
}
impl GuiState {
    fn default(presets: Vec<PathBuf>) -> GuiState {
        GuiState {
            animate: true,
            fps_text: String::new(),
            presets,
            mode: GuiMode::Transform,
            show_progress_modal: false,
            last_progress: 0.0,
            process_cancelled: Arc::new(AtomicBool::new(false)),
            quick_process: false,
            last_mouse_pos: None,
            drawing_color: [0.0, 0.0, 0.0, DRAWING_ALPHA],
            currently_processing: None,
        }
    }
}

impl App for VoronoiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        let device = &rs.device;
        // Resize handling (match the egui "central panel" size)
        //let available = ctx.available_rect();
        // let target_size = (
        //     available.width().max(1.0) as u32,
        //     available.height().max(1.0) as u32,
        // );
        // if target_size != self.size {
        //     self.resize(rs, target_size);
        // }

        // Ensure texture is registered exactly once per allocation
        self.ensure_registered_texture(rs);

        // Run GPU pipeline
        if let Some(img) = &self.preview_image {
            // show image
            let img = image::imageops::resize(
                img,
                self.size.0,
                self.size.1,
                image::imageops::FilterType::Nearest,
            );
            let rgba: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = img.convert();
            let rgba = rgba.into_raw();
            rs.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.color_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.size.0),
                    rows_per_image: Some(self.size.1),
                },
                wgpu::Extent3d {
                    width: self.size.0,
                    height: self.size.1,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            self.run_gpu(rs);

            if self.gui.animate {
                if self.gif_status.is_recording() {
                    for _ in 0..(60 / GIF_FRAMERATE) {
                        self.sim.update(&mut self.seeds, self.size.0);
                    }

                    if let Err(e) = self.write_frame_to_gif(device, &rs.queue) {
                        self.gif_status = GifStatus::Error(e.to_string());
                        self.gui.animate = false;
                    } else {
                        self.gif_frame_count += 1;

                        if self.gif_frame_count >= GIF_NUM_FRAMES {
                            // finish recording
                            // self.stop_recording_gif(device);
                            if let GifStatus::Recording(Some(path)) = self.gif_status.clone() {
                                self.gif_status = GifStatus::Complete(path);
                            } else {
                                self.gif_status = GifStatus::Error(format!(
                                    "Something weird happened: {:?}",
                                    self.gif_status
                                ));
                            }

                            self.gui.animate = false;
                        }
                    }
                } else {
                    self.sim.update(&mut self.seeds, self.size.0);
                }
                rs.queue
                    .write_buffer(&self.seed_buf, 0, bytemuck::cast_slice(&self.seeds));
            }
        }

        let dt = self.prev_frame_time.elapsed();
        self.prev_frame_time = std::time::Instant::now();
        self.gui.fps_text = format!(
            "{:5.2} ms/frame (~{:06.0} FPS)",
            dt.as_secs_f64() * 1000.0,
            1.0 / dt.as_secs_f64()
        );

        // ===== UI =====
        ctx.set_zoom_factor(1.4);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                match self.gui.mode {
                    GuiMode::Draw => {
                        if ui.button("reset").clicked() {
                            self.init_canvas(device);
                        }

                        if let Ok(msg) = self.progress_rx.try_recv() {
                            match msg {
                                ProgressMsg::UpdatePreview(image) => {
                                    self.preview_image = Some(image);
                                }
                                ProgressMsg::Cancelled => {
                                    self.gui
                                        .process_cancelled
                                        .store(false, std::sync::atomic::Ordering::Relaxed);
                                    self.preview_image = None;

                                    ui.close();
                                }
                                ProgressMsg::UpdateAssignments(assignments) => {
                                    self.sim.set_assignments(assignments, self.size.0)
                                }
                                ProgressMsg::Progress(_) => todo!(),
                                ProgressMsg::Done(path_buf) => todo!(),
                                ProgressMsg::Error(_) => todo!(),
                            }
                        }

                        if ui
                            .add(egui::Button::new(egui::RichText::new("🏠")))
                            .on_hover_text("transform mode")
                            .clicked()
                        {
                            self.gui.mode = GuiMode::Transform;
                            self.change_sim(device, self.gui.presets[0].clone());
                        }
                    }
                    GuiMode::Transform => {
                        if ui
                            .add_enabled(
                                !self.gui.animate,
                                egui::Button::new("play transformation"), //.fill(egui::Color32::from_rgb(47, 92, 34)),
                            )
                            .clicked()
                        {
                            self.gui.animate = true;
                        }
                        if ui
                            .add_enabled(self.gui.animate, egui::Button::new("switch target"))
                            .clicked()
                        {
                            self.sim.switch();
                        }
                        if ui.button("reload").clicked() {
                            self.change_sim(device, self.sim.source_path());
                            self.gui.animate = false;
                        }
                        ui.separator();

                        if ui.button("save gif").clicked() {
                            self.gif_status = GifStatus::Recording(None);
                            self.resize_textures(device, (GIF_RESOLUTION, GIF_RESOLUTION), false);
                            self.change_sim(device, self.sim.source_path());
                            self.gui.animate = true;
                            for _ in 0..20 {
                                self.sim.update(&mut self.seeds, self.size.0);
                            }
                        }

                        ui.separator();
                        // choose preset
                        // for (i, preset) in self.gui.presets.clone().into_iter().enumerate() {
                        //     if ui.button(i.to_string()).clicked() {
                        //         self.change_sim(device, preset);
                        //         self.gui.animate = false;
                        //     }
                        // }
                        ui.label("choose preset:");
                        egui::ComboBox::from_label("")
                            .selected_text({
                                let name = self.sim.name();
                                if name.chars().count() > 13 {
                                    let truncated: String = name.chars().take(10).collect();
                                    format!("{truncated}…")
                                } else {
                                    name.to_string()
                                }
                            })
                            .show_ui(ui, |ui| {
                                for preset in self.gui.presets.clone().into_iter() {
                                    if ui.button(preset_path_to_name(&preset)).clicked() {
                                        // Call change_sim when a new preset is selected
                                        self.change_sim(device, preset);
                                        self.gui.animate = false;
                                    }
                                }
                            });

                        if ui.button("obamify new image").clicked() {
                            // open file select
                            let file = rfd::FileDialog::new()
                                .set_title("choose image (square aspect ratio recommended)")
                                .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                                .pick_file();
                            if let Some(path) = file {
                                self.gui.show_progress_modal = true;
                                self.gui.quick_process = false;

                                let settings = GenerationSettings::default();
                                self.gui.currently_processing = Some(path.clone());
                                //self.change_sim(device, path.clone(), false);

                                std::thread::spawn({
                                    let tx = self.progress_tx.clone();
                                    let cancelled = self.gui.process_cancelled.clone();
                                    move || {
                                        let result = calculate::process_optimal(
                                            path,
                                            settings,
                                            tx.clone(),
                                            cancelled,
                                        );
                                        match result {
                                            Ok(()) => {}
                                            Err(err) => {
                                                tx.send(ProgressMsg::Error(err.to_string())).ok();
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        ui.separator();

                        if ui
                            .add(egui::Button::new(egui::RichText::new("✏")))
                            .on_hover_text("drawing mode")
                            .clicked()
                        {
                            self.gui.mode = GuiMode::Draw;
                            self.init_canvas(device);
                        }

                        if self.gui.show_progress_modal {
                            Window::new("progress")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .anchor(egui::Align2::CENTER_BOTTOM, (0.0, 0.0))
                                .show(ui.ctx(), |ui| {
                                    let processing_label_message = if self.gui.quick_process {
                                        "processing..."
                                    } else {
                                        "processing... (could take a while)"
                                    };
                                    ui.vertical(|ui| {
                                    ui.set_min_width(ui.available_width().min(400.0));
                                    if let Ok(msg) = self.progress_rx.try_recv() {
                                        match msg {
                                            ProgressMsg::Done(path) => {
                                                self.preview_image = None;
                                                self.gui.presets = get_presets();
                                                self.change_sim(device, path);
                                                self.gui.animate = true;
                                                self.gui.show_progress_modal = false;
                                                ui.close();
                                            }
                                            ProgressMsg::Progress(p) => {
                                                self.gui.last_progress = p;
                                            }
                                            ProgressMsg::Error(err) => {
                                                ui.label(format!("error: {}", err));
                                                if ui.button("close").clicked() {
                                                    ui.close();
                                                }
                                            }
                                            ProgressMsg::UpdatePreview(image) => {
                                                self.preview_image = Some(image);
                                            }
                                            ProgressMsg::Cancelled => {
                                                self.gui.process_cancelled.store(
                                                    false,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );
                                                self.preview_image = None;
                                                if self.gui.quick_process {
                                                    let settings = GenerationSettings::default();

                                                    std::thread::spawn({
                                                        let tx = self.progress_tx.clone();
                                                        let cancelled =
                                                            self.gui.process_cancelled.clone();
                                                        let path = self.gui
                                                            .currently_processing
                                                            .clone()
                                                            .unwrap();
                                                        move || match calculate::process_genetic(
                                                            path,
                                                            settings,
                                                            tx.clone(),
                                                            cancelled,
                                                        ) {
                                                            Ok(()) => {}
                                                            Err(err) => {
                                                                tx.send(ProgressMsg::Error(
                                                                    err.to_string(),
                                                                ))
                                                                .ok();
                                                            }
                                                        }
                                                    });
                                                } else {
                                                    self.gui.show_progress_modal = false;
                                                    ui.close();
                                                }
                                            }
                                            ProgressMsg::UpdateAssignments(assignments) => {
                                                self.sim.set_assignments(assignments, self.size.0)
                                            }
                                        }
                                    }

                                    if self.gui
                                        .process_cancelled
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                    {
                                        ui.label("cancelling...");
                                    } else {
                                        ui.label(processing_label_message);
                                    }
                                    ui.add(
                                        egui::ProgressBar::new(self.gui.last_progress)
                                            .show_percentage(),
                                    );

                                    ui.horizontal(|ui| {
                                        if ui.button("cancel").clicked() {
                                            self.gui.quick_process = false;
                                            self.gui.process_cancelled
                                                .store(true, std::sync::atomic::Ordering::Relaxed);
                                            self.gui.last_progress = 0.0;
                                        }

                                        if !self.gui.quick_process
                                            && ui
                                                .button("make faster, lower quality result instead")
                                                .clicked()
                                        {
                                            self.gui.process_cancelled
                                                .store(true, std::sync::atomic::Ordering::Relaxed);
                                            self.gui.last_progress = 0.0;
                                            self.gui.quick_process = true;
                                        }
                                    })
                                });
                                });

                            // if modal.should_close() {
                            //     self.gui.show_progress_modal = false;
                            // }
                        } else if !self.gif_status.not_recording() {
                            Modal::new("recording_progress".into()).show(ui.ctx(), |ui| match self
                                .gif_status
                                .clone()
                            {
                                GifStatus::Recording(_) => {
                                    ui.label("Recording GIF...");
                                }

                                GifStatus::Error(err) => {
                                    ui.label(format!("Error: {}", err));
                                    ui.horizontal(|ui| {
                                        if ui.button("close").clicked() {
                                            self.stop_recording_gif(device);
                                        }
                                    });
                                }
                                GifStatus::Complete(path) => {
                                    ui.label("gif saved!");
                                    ui.horizontal(|ui| {
                                        if ui.button("open file").clicked() {
                                            opener::reveal(path).ok();
                                        }
                                        if ui.button("close").clicked() {
                                            self.stop_recording_gif(device);
                                        }
                                    });
                                }
                                GifStatus::None => unreachable!(),
                            });
                        }

                        // ui.separator();
                        // ui.label(&self.gui.fps_text);
                    }
                }
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                ui.vertical_centered_justified(|ui| {
                    if let Some(id) = self.egui_tex_id {
                        let full = ui.available_size();
                        let aspect = self.size.0 as f32 / self.size.1 as f32;
                        let desired = full.x.min(full.y) * egui::vec2(1.0, aspect);
                        ui.add(egui::Image::new((id, desired)).maintain_aspect_ratio(true));

                        if matches!(self.gui.mode, GuiMode::Draw) {
                            self.handle_drawing(ctx, device, ui, aspect);
                        }
                    } else {
                        ui.colored_label(Color32::LIGHT_RED, "Texture not ready");
                    }
                });
            });
        if matches!(self.gui.mode, GuiMode::Draw) {
            let number_keys = [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
                egui::Key::Num5,
            ];

            // DBECEE,383232, 6B5E57,D49976

            let colors = [
                ("black", 0x000000),
                ("a", 0x86d9e3),
                ("b", 0x383232),
                ("c", 0xD49976),
                ("d", 0x793025),
            ];

            for (idx, (name, color)) in colors.iter().enumerate() {
                if ctx.input(|i| i.key_pressed(number_keys[idx])) {
                    let hex = *color;
                    let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
                    let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
                    let b = (hex & 0xFF) as f32 / 255.0;
                    let a = 0.5;

                    self.gui.drawing_color = [r, g, b, a];
                }
            }
            // show selected drawing color
            egui::Area::new("drawing_color".into())
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 30.0))
                .show(ctx, |ui| {
                    let rect_size = 30.0;
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(rect_size, rect_size),
                        egui::Sense::hover(),
                    );
                    let color = egui::Color32::from_rgba_unmultiplied(
                        (self.gui.drawing_color[0] * 255.0) as u8,
                        (self.gui.drawing_color[1] * 255.0) as u8,
                        (self.gui.drawing_color[2] * 255.0) as u8,
                        255,
                    );
                    ui.painter().rect_filled(rect, 15.0, color);
                    if ui.is_rect_visible(rect) {
                        ui.painter().rect_stroke(
                            rect,
                            15.0,
                            (2.0, egui::Color32::WHITE),
                            egui::StrokeKind::Inside,
                        );
                    }

                    // Keep the picker visible while hovering either the main swatch or the picker area.
                    let spacing = 10.0;
                    let btn_size = rect_size / 2.0;
                    let gap = 4.0;

                    // Layout picker row next to the swatch, vertically centered.
                    let n_buttons = colors.len() as f32;
                    let picker_width = n_buttons * btn_size + (n_buttons - 1.0).max(0.0) * gap;
                    let picker_min =
                        rect.min + egui::vec2(rect_size + spacing, (rect_size - btn_size) * 0.5);
                    let picker_rect = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(picker_width + rect_size + spacing, rect_size),
                    );

                    // Decide visibility purely from pointer position to avoid z-order flicker.
                    let pointer_pos = ctx.input(|i| i.pointer.hover_pos());
                    let show_picker = ui.is_rect_visible(rect)
                        && pointer_pos.is_some_and(|p| rect.contains(p) || picker_rect.contains(p));

                    // Global visibility animation driver
                    let base_t = ui
                        .ctx()
                        .animate_bool(egui::Id::new("color_picker_visible"), show_picker);

                    // Helpers
                    let saturate = |x: f32| x.clamp(0.0, 1.0);
                    let smoothstep = |x: f32| {
                        let x = saturate(x);
                        x * x * (3.0 - 2.0 * x)
                    };

                    // Start position = centered under the main swatch so buttons "emerge" from it
                    let start_pos = egui::pos2(
                        rect.min.x + (rect_size - btn_size) * 0.5,
                        rect.min.y + (rect_size - btn_size) * 0.5,
                    );

                    // Per-button stagger
                    let per_btn_delay = 0.08_f32;
                    // Ensure the last button still reaches t=1 when base_t=1
                    let total_stagger = (n_buttons - 1.0).max(0.0) * per_btn_delay;
                    let denom = (1.0 - total_stagger).max(1e-6);

                    for (idx, (_name, hex)) in colors.iter().enumerate() {
                        let rgba = {
                            let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
                            let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
                            let b = (hex & 0xFF) as f32 / 255.0;
                            let a = DRAWING_ALPHA;
                            [r, g, b, a]
                        };
                        let i = idx as f32;

                        // Staggered progress for each button; normalized so the last also reaches 1.0
                        let raw = (base_t - per_btn_delay * i) / denom;
                        let t_i = smoothstep(raw);

                        // Only draw while animating or visible to avoid early reveal
                        if t_i <= 0.001 {
                            continue;
                        }

                        // Target position to the right of the swatch
                        let end_pos = egui::pos2(picker_min.x + i * (btn_size + gap), picker_min.y);

                        // Interpolate from under the swatch to the target
                        let pos = egui::pos2(
                            egui::lerp(start_pos.x..=end_pos.x, t_i),
                            egui::lerp(start_pos.y..=end_pos.y, t_i),
                        );

                        egui::Area::new(egui::Id::new(format!("color_picker_btn_{idx}")))
                            .fixed_pos(pos)
                            .show(ctx, |ui| {
                                let (btn_rect, btn_resp) = ui.allocate_exact_size(
                                    egui::vec2(btn_size, btn_size),
                                    egui::Sense::click(),
                                );

                                // Fade with the slide
                                let a = (255.0 * t_i) as u8;
                                let color32 = egui::Color32::from_rgba_unmultiplied(
                                    (rgba[0] * 255.0) as u8,
                                    (rgba[1] * 255.0) as u8,
                                    (rgba[2] * 255.0) as u8,
                                    a,
                                );

                                ui.painter().rect_filled(btn_rect, 15.0 / 2.0, color32);
                                if ui.is_rect_visible(btn_rect) {
                                    ui.painter().rect_stroke(
                                        btn_rect,
                                        15.0 / 2.0,
                                        (
                                            2.0,
                                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, a),
                                        ),
                                        egui::StrokeKind::Inside,
                                    );
                                }

                                if btn_resp.clicked() {
                                    self.gui.drawing_color = rgba;
                                }
                            });
                    }
                });
        }

        // continuous repaint for animation
        ctx.request_repaint();
        self.frame_count += 1;
    }
}

fn main() -> eframe::Result<()> {
    // get screen dimensions
    let native_options = NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size((1024.0, 1024.0))
            .with_clamp_size_to_monitor_size(true)
            .with_resizable(true),
        ..Default::default()
    }; // wgpu backend is selected via the `wgpu` feature
    eframe::run_native(
        "obamify drawing tool",
        native_options,
        Box::new(|cc| Ok(Box::new(VoronoiApp::new(cc)))),
    )
}
