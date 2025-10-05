mod calculate;
mod gif_recorder;
mod gui;
mod morph_sim;
mod preset;

#[cfg(target_arch = "wasm32")]
pub use crate::app::calculate::worker::worker_entry;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::{
    num::NonZeroU64,
    sync::{Arc, RwLock, atomic::AtomicU32},
};

use bytemuck::{Pod, Zeroable};
use eframe::CreationContext;
use egui_wgpu::{self, wgpu};
use uuid::Uuid;
use wgpu::util::DeviceExt;

const WG_SIZE_XY: u32 = 8;
const WG_SIZE_SEEDS: u32 = 256;
//const INVALID_ID: u32 = 0xFFFF_FFFF;

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
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_RESOLUTION: u32 = 2048;

#[cfg(target_arch = "wasm32")]
const DEFAULT_RESOLUTION: u32 = 1024;

pub enum GuiMode {
    Transform,
    Draw,
}

use crate::app::{calculate::ProgressMsg, morph_sim::Sim, preset::UnprocessedPreset};
use crate::app::{
    calculate::{GenerationSettings, drawing_process::PixelData},
    preset::Preset,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::closure::Closure;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use web_sys::{Worker, WorkerOptions, WorkerType, js_sys};

pub struct ObamifyApp {
    //prev_frame_time: std::time::Instant,
    // UI state
    size: (u32, u32),
    seed_count: u32,

    #[cfg(not(target_arch = "wasm32"))]
    progress_tx: mpsc::SyncSender<ProgressMsg>,
    #[cfg(not(target_arch = "wasm32"))]
    progress_rx: mpsc::Receiver<ProgressMsg>,

    #[cfg(target_arch = "wasm32")]
    worker: Option<Worker>,

    #[cfg(target_arch = "wasm32")]
    inbox: Vec<ProgressMsg>,

    gif_recorder: gif_recorder::GifRecorder,
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

    gui: gui::GuiState,
    current_drawing_id: Arc<AtomicU32>,
}

impl ObamifyApp {
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

    pub fn change_sim(&mut self, device: &wgpu::Device, source: Preset, change_index: usize) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_image(self.size.0, source);
        self.apply_sim_init(device, seed_count, seeds, colors, sim);
        self.gui.current_preset = change_index;
    }

    pub fn canvas_sim(&mut self, device: &wgpu::Device, source: &UnprocessedPreset) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_canvas(self.size.0, source.clone());
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
        let presets: Vec<Preset> = if let Some(storage) = cc.storage {
            eframe::get_value(storage, "presets").unwrap_or(get_presets())
        } else {
            get_presets()
        };

        #[cfg(target_arch = "wasm32")]
        let random_preset = (js_sys::Math::random() * (presets.len() as f64)) as usize;

        #[cfg(not(target_arch = "wasm32"))]
        let random_preset = frand::Rand::with_seed(
            std::time::SystemTime::now().elapsed().unwrap().as_nanos() as u64,
        )
        .gen_range(0..presets.len() as u64) as usize;

        let (seed_count, seeds, colors, sim) =
            morph_sim::init_image(size.0, presets[random_preset].clone());

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

        #[cfg(not(target_arch = "wasm32"))]
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
            //prev_frame_time: std::time::Instant::now(),
            #[cfg(not(target_arch = "wasm32"))]
            progress_tx,
            #[cfg(not(target_arch = "wasm32"))]
            progress_rx,
            gif_recorder: gif_recorder::GifRecorder::new(),
            preview_image: None,

            stroke_count: 0,
            gui: gui::GuiState::default(presets, random_preset),
            frame_count: 0,
            current_drawing_id: Arc::new(AtomicU32::new(0)),
            #[cfg(target_arch = "wasm32")]
            worker: None,
            #[cfg(target_arch = "wasm32")]
            inbox: Vec::new(),
        }
    }

    pub fn get_latest_msg(&mut self) -> Option<ProgressMsg> {
        #[cfg(target_arch = "wasm32")]
        {
            self.inbox.pop()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            match self.progress_rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("progress channel disconnected");
                    None
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn ensure_worker(&mut self, ctx: &egui::Context) {
        if self.worker.is_some() {
            return;
        }

        let worker = {
            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let w = Worker::new_with_options("worker.js", &opts).expect("worker");

            // ---- onerror: may be ErrorEvent OR a generic Event/JsValue ----
            let onerror = Closure::wrap(Box::new(move |e: JsValue| {
                if let Some(err) = e.dyn_ref::<web_sys::ErrorEvent>() {
                    // Safe: has .message()
                    web_sys::console::error_2(&"worker error:".into(), &err.message().into());
                    // (Optional) filenames/lineno may be empty on module workers:
                    // web_sys::console::error_3(&"at".into(), &err.filename().into(), &err.lineno().into());
                } else if let Some(ev) = e.dyn_ref::<web_sys::Event>() {
                    // No message property
                    let ty = ev.type_();
                    web_sys::console::error_2(&"worker error (generic Event):".into(), &ty.into());
                } else {
                    // Something else (could even be undefined/null)
                    web_sys::console::error_1(&JsValue::from_str(&format!(
                        "worker error (unknown): {:?}",
                        js_sys::JSON::stringify(&e).ok()
                    )));
                }
            }) as Box<dyn FnMut(JsValue)>);
            // set_onerror takes a Function; unchecked_ref is fine here
            w.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onerror.forget();

            // ---- onmessageerror: data failed to deserialize ----
            let onmsgerr = Closure::wrap(Box::new(move |e: JsValue| {
                if let Some(me) = e.dyn_ref::<web_sys::MessageEvent>() {
                    web_sys::console::error_2(&"worker messageerror; data:".into(), &me.data());
                } else {
                    web_sys::console::error_1(&"worker messageerror (unknown payload)".into());
                }
            }) as Box<dyn FnMut(JsValue)>);
            // Older web-sys may not have set_onmessageerror; ignore if missing
            #[allow(unused_must_use)]
            {
                w.set_onmessageerror(Some(onmsgerr.as_ref().unchecked_ref()));
            }
            onmsgerr.forget();

            w
        };

        //web_sys::console::log_1(&"worker created".into());

        // Receive progress messages
        {
            let inbox_ptr: *mut Vec<ProgressMsg> = &mut self.inbox;
            let onmessage = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                if let Ok(msg) = serde_wasm_bindgen::from_value::<ProgressMsg>(e.data()) {
                    // SAFETY: single-threaded; worker posts to main thread
                    unsafe {
                        (*inbox_ptr).push(msg);
                    }
                }
            }) as Box<dyn FnMut(_)>);
            worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget();
        }

        self.worker = Some(worker);
    }

    #[cfg(target_arch = "wasm32")]
    fn start_job(&mut self, src: UnprocessedPreset, settings: GenerationSettings) {
        if let Some(w) = &self.worker {
            let req = calculate::worker::WorkerReq::Process {
                source: src,
                settings,
            };
            let v = serde_wasm_bindgen::to_value(&req).unwrap();
            w.post_message(&v).unwrap();
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

    fn stop_recording_gif(&mut self, device: &wgpu::Device) {
        self.gif_recorder.stop();
        self.gui.animate = false;
        self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION), false);
        self.reset_sim(device);
    }

    fn reset_sim(&mut self, device: &wgpu::Device) {
        self.change_sim(
            device,
            self.gui.presets[self.gui.current_preset].clone(),
            self.gui.current_preset,
        );
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

    #[cfg(not(target_arch = "wasm32"))]
    fn init_canvas(&mut self, device: &wgpu::Device) {
        let blank = image::load_from_memory(include_bytes!("./app/calculate/blank.png"))
            .unwrap()
            .to_rgba8();

        let settings = GenerationSettings::default(Uuid::new_v4(), "canvas".to_string());
        let source = UnprocessedPreset {
            name: "canvas".to_string(),
            width: blank.width(),
            height: blank.height(),
            source_img: blank.into_raw(),
        };
        self.canvas_sim(device, &source);
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
            let source = source.clone();
            move || {
                let result = calculate::drawing_process::drawing_process_genetic(
                    source,
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
        (px - x0).hypot(py - y0)
    } else {
        // Calculate the t that minimizes the distance.
        let t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy);
        if t < 0.0 {
            // Beyond the 'x0,y0' end of the segment
            (px - x0).hypot(py - y0)
        } else if t > 1.0 {
            // Beyond the 'x1,y1' end of the segment
            (px - x1).hypot(py - y1)
        } else {
            // Projection falls on the segment
            let proj_x = x0 + t * dx;
            let proj_y = y0 + t * dy;
            (px - proj_x).hypot(py - proj_y)
        }
    }
}

macro_rules! include_presets {
    ($($name:literal),*) => {
        fn get_presets() -> Vec<Preset> {
            vec![
                $({
                    let img = image::load_from_memory(include_bytes!(concat!(
                        "../presets/",
                        $name,
                        "/source.png"
                    )))
                    .unwrap()
                    .to_rgb8();
                    Preset {
                        inner: UnprocessedPreset {
                            name: $name.to_owned(),
                            width: img.width(),
                            height: img.height(),
                            source_img: img.into_raw(),
                        },
                        assignments: include_str!(concat!("../presets/", $name, "/assignments.json"))
                            .to_string()
                            .strip_prefix('[')
                            .unwrap()
                            .strip_suffix(']')
                            .unwrap()
                            .split(',')
                            .map(|s| s.parse().unwrap())
                            .collect::<Vec<usize>>(),
                    }
                }),*
            ]
        }
    };
}

include_presets! { "wisetree", "blackhole", "cat", "cat2", "colorful" }
