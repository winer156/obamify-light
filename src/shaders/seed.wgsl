@group(0) @binding(0) var seed_tex: texture_2d<f32>;

struct ParamsCommon { width: u32, height: u32, n_seeds: u32, _pad: u32 };
@group(0) @binding(1) var<uniform> params: ParamsCommon;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) seed_id: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var output: VertexOutput;
  let seed_id = vertex_index;
  
  if (seed_id >= params.n_seeds) {
    output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    output.seed_id = 0u;
    return output;
  }
  
  let tex_width = 1024u;
  let seed_x = seed_id % tex_width;
  let seed_y = seed_id / tex_width;
  let seed_uv = vec2<i32>(i32(seed_x), i32(seed_y));
  let p = textureLoad(seed_tex, seed_uv, 0).rg;
  
  let x = ((p.x + 0.5) / f32(params.width)) * 2.0 - 1.0;
  let y = ((p.y + 0.5) / f32(params.height)) * 2.0 - 1.0;
  output.position = vec4<f32>(x, -y, 0.0, 1.0);
  output.seed_id = seed_id;
  return output;
}

@fragment
fn fs_main(@location(0) seed_id: u32) -> @location(0) vec4<f32> {
  let r = f32((seed_id >> 0u) & 0xFFu) / 255.0;
  let g = f32((seed_id >> 8u) & 0xFFu) / 255.0;
  let b = f32((seed_id >> 16u) & 0xFFu) / 255.0;
  let a = f32((seed_id >> 24u) & 0xFFu) / 255.0;
  return vec4<f32>(r, g, b, a);
}