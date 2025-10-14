@group(0) @binding(0) var ids: texture_2d<f32>;
@group(0) @binding(1) var ids_sampler: sampler;

@group(0) @binding(2) var seed_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;

struct ParamsCommon { width: u32, height: u32, n_seeds: u32, _pad: u32 };
@group(0) @binding(4) var<uniform> params: ParamsCommon;

fn load_seed_pos(seed_id: u32) -> vec2<f32> {
    let tex_width = 1024u;
    let seed_x = seed_id % tex_width;
    let seed_y = seed_id / tex_width;
    return textureLoad(seed_tex, vec2<i32>(i32(seed_x), i32(seed_y)), 0).rg;
}

fn load_color(seed_id: u32) -> vec4<f32> {
    let tex_width = 1024u;
    let color_x = seed_id % tex_width;
    let color_y = seed_id / tex_width;
    return textureLoad(color_tex, vec2<i32>(i32(color_x), i32(color_y)), 0);
}

fn decode_id(rgba: vec4<f32>) -> u32 {
    let r = u32(rgba.r * 255.0 + 0.5);
    let g = u32(rgba.g * 255.0 + 0.5);
    let b = u32(rgba.b * 255.0 + 0.5);
    let a = u32(rgba.a * 255.0 + 0.5);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn dist2(a: vec2<f32>, b: vec2<f32>) -> f32 { let d = a - b; return dot(d,d); }

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var output: VertexOutput;
  let x = f32((vertex_index & 1u) << 1u) - 1.0;
  let y = f32((vertex_index & 2u)) - 1.0;
  output.position = vec4<f32>(x, -y, 0.0, 1.0);
  output.uv = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);
  return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let gid = vec2<i32>(i32(uv.x * f32(params.width)), i32(uv.y * f32(params.height)));
  
  if (gid.x >= i32(params.width) || gid.y >= i32(params.height)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  
  let id_rgba = textureLoad(ids, gid, 0);
  let id = decode_id(id_rgba);
  let seed = load_seed_pos(id);
//   if dist2(seed, vec2<f32>(f32(gid.x), f32(gid.y))) < 10.0 {
//     // draw seed position in white
//     return vec4<f32>(1.0,1.0,1.0,1.0);
//   }
  var rgba: vec4<f32>;
  if (id == 0xfffffffFu) {
    rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  } else {
    rgba = load_color(id);
  }
  return rgba;
}