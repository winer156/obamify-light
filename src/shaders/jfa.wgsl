@group(0) @binding(0) var seed_tex: texture_2d<f32>;

@group(0) @binding(1) var src_ids: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct JfaParams {
    width: u32,
    height: u32,
    step: u32,
    _pad: u32
};

@group(0) @binding(3) var<uniform> params: JfaParams;

fn load_seed_pos(seed_id: u32) -> vec2<f32> {
    let tex_width = 1024u;
    let seed_x = seed_id % tex_width;
    let seed_y = seed_id / tex_width;
    return textureLoad(seed_tex, vec2<i32>(i32(seed_x), i32(seed_y)), 0).rg;
}

fn dist2(a: vec2<f32>, b: vec2<f32>) -> f32 { let d = a - b; return dot(d,d); }

fn decode_id(rgba: vec4<f32>) -> u32 {
    let r = u32(rgba.r * 255.0 + 0.5);
    let g = u32(rgba.g * 255.0 + 0.5);
    let b = u32(rgba.b * 255.0 + 0.5);
    let a = u32(rgba.a * 255.0 + 0.5);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

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
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let p = vec2<f32>(f32(gid.x), f32(gid.y));
    let center_rgba = textureLoad(src_ids, gid, 0);
    var best_id: u32 = decode_id(center_rgba);
    var best_d2: f32 = 3.4e38;
    if (best_id != 0xfffffffFu) { best_d2 = dist2(p, load_seed_pos(best_id)); }

    let s = i32(params.step);
    let offs = array<vec2<i32>,8>(
        vec2<i32>( s, 0), vec2<i32>(-s, 0), vec2<i32>( 0, s), vec2<i32>( 0,-s),
        vec2<i32>( s, s), vec2<i32>( s,-s), vec2<i32>(-s, s), vec2<i32>(-s,-s)
    );

    for (var i = 0; i < 8; i = i + 1) {
        let q = gid + offs[i];
        if (q.x < 0 || q.y < 0 || q.x >= i32(params.width) || q.y >= i32(params.height)) {
            continue;
        }
        
        let cand_rgba = textureLoad(src_ids, q, 0);
        let cand = decode_id(cand_rgba);
        if (cand != 0xfffffffFu) {
            let d2 = dist2(p, load_seed_pos(cand));
            if (d2 < best_d2) { best_d2 = d2; best_id = cand; }
        }
    }

    let r = f32((best_id >> 0u) & 0xFFu) / 255.0;
    let g = f32((best_id >> 8u) & 0xFFu) / 255.0;
    let b = f32((best_id >> 16u) & 0xFFu) / 255.0;
    let a = f32((best_id >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}