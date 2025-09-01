@group(0) @binding(0) var ids: texture_2d<u32>;
@group(0) @binding(1) var out_color: texture_storage_2d<rgba8unorm, write>;
struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(2) var<storage, read> seeds: Seeds;

struct Colors { rgba: array<vec4<f32>> };
@group(0) @binding(3) var<storage, read> colors: Colors;

fn hash32(x: u32) -> u32 {
  var v = x; // xorshift* style
  v ^= v << 13u; v ^= v >> 17u; v ^= v << 5u;
  return v * 0x9E3779B9u;
}

fn dist2(a: vec2<f32>, b: vec2<f32>) -> f32 { let d = a - b; return dot(d,d); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_color);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let id = textureLoad(ids, vec2<i32>(gid.xy), 0).r;
  let seed = seeds.pos[id];
//   if dist2(seed.xy, vec2<f32>(f32(gid.x), f32(gid.y))) < 10.0 {
//     // draw seed position in white
//     textureStore(out_color, vec2<i32>(gid.xy), vec4<f32>(1.0,1.0,1.0,1.0));
//     return;
//   }
  var rgba: vec4<f32>;
  if (id == 0xfffffffFu) {
    rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  } else {
    rgba = colors.rgba[id];
  }
  textureStore(out_color, vec2<i32>(gid.xy), rgba);
}