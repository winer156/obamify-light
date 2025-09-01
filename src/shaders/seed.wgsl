struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(0) var<storage, read> seeds: Seeds;

struct ParamsCommon { width: u32, height: u32, n_seeds: u32, _pad: u32 };
@group(0) @binding(1) var<uniform> params: ParamsCommon;

@group(0) @binding(2) var dst_ids: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_seeds) { return; }
  let p = seeds.pos[i];
  let x = i32(round(p.x));
  let y = i32(round(p.y));
  textureStore(dst_ids, vec2<i32>(x,y), vec4<u32>(i, 0u, 0u, 0u));
}