use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub inner: UnprocessedPreset,
    pub assignments: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UnprocessedPreset {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub source_img: Vec<u8>,
}
