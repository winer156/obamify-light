struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(0) var<storage, read> seeds: Seeds;

@group(0) @binding(1) var src_ids: texture_2d<u32>;
@group(0) @binding(2) var dst_ids: texture_storage_2d<r32uint, write>;

struct JfaParams {
    width: u32,
    height: u32,
    step: u32,
    _pad: u32
};

@group(0) @binding(3) var<uniform> params: JfaParams;

fn dist2(a: vec2<f32>, b: vec2<f32>) -> f32 { let d = a - b; return dot(d,d); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let p = vec2<f32>(f32(gid.x), f32(gid.y));
    var best_id: u32 = textureLoad(src_ids, vec2<i32>(gid.xy), 0).r;
    var best_d2: f32 = 3.4e38;
    if (best_id != 0xfffffffFu) { best_d2 = dist2(p, seeds.pos[best_id]); }

    let s = i32(params.step);
    // 8-neighborhood at distance 'step'
    let offs = array<vec2<i32>,8>(
        vec2<i32>( s, 0), vec2<i32>(-s, 0), vec2<i32>( 0, s), vec2<i32>( 0,-s),
        vec2<i32>( s, s), vec2<i32>( s,-s), vec2<i32>(-s, s), vec2<i32>(-s,-s)
    );

    for (var i = 0; i < 8; i = i + 1) {
        let q = vec2<i32>(gid.xy) + offs[i];
        if (q.x < 0 || q.y < 0 || q.x >= i32(params.width) || q.y >= i32(params.height)) {
            continue;
        }
        
        let cand = textureLoad(src_ids, q, 0).r;
        if (cand != 0xfffffffFu) {
            let d2 = dist2(p, seeds.pos[cand]);
            if (d2 < best_d2) { best_d2 = d2; best_id = cand; }
        }
    }

    textureStore(dst_ids, vec2<i32>(gid.xy), vec4<u32>(best_id, 0u, 0u, 0u));
}