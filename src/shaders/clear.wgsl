@group(0) @binding(0) var dst_ids: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(dst_ids);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  textureStore(dst_ids, vec2<i32>(gid.xy), vec4<u32>(0xfffffffFu, 0u, 0u, 0u));
}