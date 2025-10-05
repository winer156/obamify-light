#![allow(clippy::type_complexity)]

use image::ImageBuffer;

use super::load_weights;

use super::GenerationSettings;

use std::error::Error;

// pub(crate) fn save_result(
//     target: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
//     base_name: String,
//     source: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
//     assignments: Vec<usize>,
//     img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
// ) -> Result<String, Box<dyn Error>> {
//     let mut dir_name = base_name.clone();
//     let mut counter = 1;
//     while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
//         dir_name = format!("{}_{}", base_name, counter);
//         counter += 1;
//     }
//     std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
//     img.save(format!("./presets/{}/output.png", dir_name))?;
//     source.save(format!("./presets/{}/source.png", dir_name))?;
//     target.save(format!("./presets/{}/target.png", dir_name))?;
//     std::fs::write(
//         format!("./presets/{}/assignments.json", dir_name),
//         serialize_assignments(assignments),
//     )?;
//     Ok(dir_name)
// }

pub(crate) fn get_images(
    source: ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    let (target, target_pixels, weights) = load_target(settings)?;

    let source = image::imageops::resize(
        &source,
        target.width(),
        target.height(),
        image::imageops::FilterType::Lanczos3,
    );
    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    Ok((target, source, source_pixels, target_pixels, weights))
}

pub(crate) fn load_target(
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    // let mut target = image::open(TARGET_IMAGE_PATH)?.to_rgb8();
    // let mut target_weights = image::open(TARGET_WEIGHTS_PATH)?.to_rgb8();
    let mut target = image::load_from_memory(include_bytes!("target.png"))?.to_rgb8();
    let mut target_weights = image::load_from_memory(include_bytes!("weights.png"))?.to_rgb8();

    if target.dimensions().0 != target.dimensions().1 {
        return Err("Target image must be square".into());
    }
    if target.dimensions() != target_weights.dimensions() {
        return Err("Target and weights images must have the same dimensions".into());
    }
    if let Some(rescale) = settings.rescale {
        target = image::imageops::resize(
            &target,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
        target_weights = image::imageops::resize(
            &target_weights,
            rescale,
            rescale,
            image::imageops::FilterType::Lanczos3,
        );
    }
    let target_pixels = target.pixels().map(|p| (p[0], p[1], p[2])).collect();
    let weights = load_weights(&target_weights.into());
    Ok((target, target_pixels, weights))
}
