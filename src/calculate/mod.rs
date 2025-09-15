use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use egui::ahash::AHasher;
use image::GenericImageView;
use pathfinding::prelude::Weights;
use rand::Rng;
use std::sync::mpsc;

use crate::{morph_sim::DRAWING_CANVAS_SIZE, PixelData, SeedColor};

pub struct GenerationSettings {
    proximity_importance: i64,
    rescale: Option<u32>,
}

impl GenerationSettings {
    pub fn default() -> Self {
        Self {
            proximity_importance: 10, // 20
            rescale: None,
        }
    }

    pub fn quick_process() -> Self {
        Self {
            proximity_importance: 10,
            rescale: Some(64),
        }
    }
}

#[inline(always)]
fn heuristic(
    apos: (u16, u16),
    bpos: (u16, u16),
    a: (u8, u8, u8),
    b: (u8, u8, u8),
    color_weight: i64,
    spatial_weight: i64,
) -> i64 {
    let spatial = (apos.0 as i64 - bpos.0 as i64).pow(2) + (apos.1 as i64 - bpos.1 as i64).pow(2);
    let color = (a.0 as i64 - b.0 as i64).pow(2)
        + (a.1 as i64 - b.1 as i64).pow(2)
        + (a.2 as i64 - b.2 as i64).pow(2);
    color * color_weight + (spatial * spatial_weight).pow(2)
}

struct ImgDiffWeights {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
    settings: GenerationSettings,
}

const TARGET_IMAGE_PATH: &str = "./target.png";
const TARGET_WEIGHTS_PATH: &str = "./weights.png";

impl Weights<i64> for ImgDiffWeights {
    fn rows(&self) -> usize {
        self.target.len()
    }

    fn columns(&self) -> usize {
        self.source.len()
    }

    #[inline(always)]
    fn at(&self, row: usize, col: usize) -> i64 {
        let (x1, y1) = (row % self.sidelen, row / self.sidelen);
        let (x2, y2) = (col % self.sidelen, col / self.sidelen);
        let (r1, g1, b1) = self.target[row];
        let (r2, g2, b2) = self.source[col];
        let weight = self.weights[row];
        -heuristic(
            (x1 as u16, y1 as u16),
            (x2 as u16, y2 as u16),
            (r1, g1, b1),
            (r2, g2, b2),
            weight,
            self.settings.proximity_importance,
        )
    }

    fn neg(&self) -> Self {
        todo!()
    }
}

pub enum ProgressMsg {
    Progress(f32),
    UpdatePreview(image::ImageBuffer<image::Rgb<u8>, Vec<u8>>),
    UpdateAssignments(Vec<usize>),
    Done(PathBuf), // result directory
    Error(String),
    Cancelled,
}

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<AHasher>>;

pub fn process_optimal<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let start_time = std::time::Instant::now();
    let (target, base_name, source, source_pixels, target_pixels, weights) =
        get_images(source_path, &settings)?;

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target_pixels,
        weights,
        sidelen: target.width() as usize,
        settings,
    };
    // pathfinding::kuhn_munkres, inlined to allow for progress bar and cancelling
    let (_total_diff, assignments) = {
        // We call x the rows and y the columns. (nx, ny) is the size of the matrix.
        let nx = weights.rows();
        let ny = weights.columns();
        assert!(
            nx <= ny,
            "number of rows must not be larger than number of columns"
        );
        // xy represents matching for x, yz matching for y
        let mut xy: Vec<Option<usize>> = vec![None; nx];
        let mut yx: Vec<Option<usize>> = vec![None; ny];
        // lx is the labelling for x nodes, ly the labelling for y nodes. We start
        // with an acceptable labelling with the maximum possible values for lx
        // and 0 for ly.
        let mut lx: Vec<i64> = (0..nx)
            .map(|row| (0..ny).map(|col| weights.at(row, col)).max().unwrap())
            .collect::<Vec<_>>();
        let mut ly: Vec<i64> = vec![0; ny];
        // s, augmenting, and slack will be reset every time they are reused. augmenting
        // contains Some(prev) when the corresponding node belongs to the augmenting path.
        let mut s = FxIndexSet::<usize>::default();
        let mut alternating = Vec::with_capacity(ny);
        let mut slack = vec![0; ny];
        let mut slackx = Vec::with_capacity(ny);
        for root in 0..nx {
            alternating.clear();
            alternating.resize(ny, None);
            // Find y such that the path is augmented. This will be set when breaking for the
            // loop below. Above the loop is some code to initialize the search.
            let mut y = {
                s.clear();
                s.insert(root);
                // Slack for a vertex y is, initially, the margin between the
                // sum of the labels of root and y, and the weight between root and y.
                // As we add x nodes to the alternating path, we update the slack to
                // represent the smallest margin between one of the x nodes and y.
                for y in 0..ny {
                    slack[y] = lx[root] + ly[y] - weights.at(root, y);
                }
                slackx.clear();
                slackx.resize(ny, root);
                Some(loop {
                    let mut delta = pathfinding::num_traits::Bounded::max_value();
                    let mut x = 0;
                    let mut y = 0;
                    // Select one of the smallest slack delta and its edge (x, y)
                    // for y not in the alternating path already.
                    for yy in 0..ny {
                        if alternating[yy].is_none() && slack[yy] < delta {
                            delta = slack[yy];
                            x = slackx[yy];
                            y = yy;
                        }
                    }
                    // If some slack has been found, remove it from x nodes in the
                    // alternating path, and add it to y nodes in the alternating path.
                    // The slack of y nodes outside the alternating path will be reduced
                    // by this minimal slack as well.
                    if delta > 0 {
                        for &x in &s {
                            lx[x] = lx[x] - delta;
                        }
                        for y in 0..ny {
                            if alternating[y].is_some() {
                                ly[y] = ly[y] + delta;
                            } else {
                                slack[y] = slack[y] - delta;
                            }
                        }
                    }
                    // Add (x, y) to the alternating path.
                    alternating[y] = Some(x);
                    if yx[y].is_none() {
                        // We have found an augmenting path.
                        break y;
                    }
                    // This y node had a predecessor, add it to the set of x nodes
                    // in the augmenting path.
                    let x = yx[y].unwrap();
                    s.insert(x);
                    // Update slack because of the added vertex in s might contain a
                    // greater slack than with previously inserted x nodes in the augmenting
                    // path.
                    for y in 0..ny {
                        if alternating[y].is_none() {
                            let alternate_slack = lx[x] + ly[y] - weights.at(x, y);
                            if slack[y] > alternate_slack {
                                slack[y] = alternate_slack;
                                slackx[y] = x;
                            }
                        }
                    }
                })
            };
            // Inverse edges along the augmenting path.
            while y.is_some() {
                let x = alternating[y.unwrap()].unwrap();
                let prec = xy[x];
                yx[y.unwrap()] = Some(x);
                xy[x] = y;
                y = prec;
            }
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                tx.send(ProgressMsg::Cancelled).unwrap();
                return Ok(());
            }
            tx.send(ProgressMsg::Progress(root as f32 / nx as f32))?;
        }
        (
            lx.into_iter().sum::<i64>() + ly.into_iter().sum::<i64>(),
            xy.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
        )
    };

    let img = make_new_img(&source_pixels, &assignments, target.width());

    let dir_name = save_result(target, base_name, source, assignments, img)?;

    tx.send(ProgressMsg::Done(PathBuf::from(format!(
        "./presets/{}",
        dir_name
    ))))?;

    println!(
        "finished in {:.2?} seconds",
        std::time::Instant::now().duration_since(start_time)
    );

    Ok(())
}

fn make_new_img(
    source_pixels: &[(u8, u8, u8)],
    assignments: &[usize],
    sidelen: u32,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let mut img = image::ImageBuffer::new(sidelen, sidelen);

    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (x, y) = (
            (target_idx % sidelen as usize) as u32,
            (target_idx / sidelen as usize) as u32,
        );
        let (r, g, b) = source_pixels[*source_idx];
        img.put_pixel(x, y, image::Rgb([r, g, b]));
    }
    img
}

fn save_result(
    target: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    base_name: String,
    source: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    assignments: Vec<usize>,
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<String, Box<dyn Error>> {
    let mut dir_name = base_name.clone();
    let mut counter = 1;
    while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
        dir_name = format!("{}_{}", base_name, counter);
        counter += 1;
    }
    std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
    img.save(format!("./presets/{}/output.png", dir_name))?;
    source.save(format!("./presets/{}/source.png", dir_name))?;
    target.save(format!("./presets/{}/target.png", dir_name))?;
    std::fs::write(
        format!("./presets/{}/assignments.json", dir_name),
        serialize_assignments(assignments),
    )?;
    Ok(dir_name)
}

fn get_images<P: AsRef<Path>>(
    source_path: P,
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        String,
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    let (target, target_pixels, weights) = load_target(settings)?;

    let base_name = source_path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let source = image::open(source_path)?.to_rgb8();
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
    Ok((
        target,
        base_name,
        source,
        source_pixels,
        target_pixels,
        weights,
    ))
}

fn load_target(
    settings: &GenerationSettings,
) -> Result<
    (
        image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        Vec<(u8, u8, u8)>,
        Vec<i64>,
    ),
    Box<dyn Error>,
> {
    let mut target = image::open(TARGET_IMAGE_PATH)?.to_rgb8();
    let mut target_weights = image::open(TARGET_WEIGHTS_PATH)?.to_rgb8();
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

#[derive(Clone, Copy)]
struct Pixel {
    src_x: u16,
    src_y: u16,
    h: i64, // current heuristic value
}

impl Pixel {
    fn new(src_x: u16, src_y: u16, h: i64) -> Self {
        Self { src_x, src_y, h }
    }

    fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    fn calc_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        colors: &[SeedColor],
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            {
                let rgba =
                    colors[self.src_y as usize * DRAWING_CANVAS_SIZE + self.src_x as usize].rgba;
                (
                    (rgba[0] * 256.0) as u8,
                    (rgba[1] * 256.0) as u8,
                    (rgba[2] * 256.0) as u8,
                )
            },
            target_col,
            weight,
            proximity_importance,
        )
    }
}

const STROKE_REWARD: i64 = -10000000000;

fn stroke_reward(
    newpos: usize,
    oldpos: usize,
    pixel_data: &[PixelData],
    pixels: &[Pixel],
    frame_count: u32,
) -> i64 {
    let x = (newpos % DRAWING_CANVAS_SIZE) as u16;
    let y = (newpos / DRAWING_CANVAS_SIZE) as u16;
    // look at 8-connected neighbors
    // if any has the same stroke_id, return true
    let data = pixel_data
        [pixels[oldpos].src_x as usize + pixels[oldpos].src_y as usize * DRAWING_CANVAS_SIZE];
    let stroke_id = data.stroke_id;
    let age = frame_count - data.last_edited;

    for (dx, dy) in [
        //(-1, -1),
        (0, -1),
        //(1, -1),
        (-1, 0),
        (1, 0),
        //(-1, 1),
        (0, 1),
        //(1, 1),
    ] {
        let nx = x as i16 + dx;
        let ny = y as i16 + dy;
        if nx < 0 || nx >= DRAWING_CANVAS_SIZE as i16 || ny < 0 || ny >= DRAWING_CANVAS_SIZE as i16
        {
            continue;
        }
        let npos = ny as usize * DRAWING_CANVAS_SIZE + nx as usize;
        if pixel_data
            [pixels[npos].src_x as usize + pixels[npos].src_y as usize * DRAWING_CANVAS_SIZE]
            .stroke_id
            == stroke_id
        {
            return STROKE_REWARD;
        }
    }
    0
}

const GENERATIONS: usize = 100;
const SWAPS_PER_GENERATION: usize = 130000;
const MAX_STROKES: usize = 2048;

pub fn process_genetic<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
    colors: Arc<std::sync::RwLock<Vec<crate::SeedColor>>>,
    pixel_data: Arc<std::sync::RwLock<Vec<PixelData>>>,
    frame_count: u32,
) -> Result<(), Box<dyn Error>> {
    let (target, base_name, source, source_pixels, target_pixels, weights) =
        get_images(source_path, &settings)?;

    let mut pixels = {
        let read_colors: Vec<SeedColor> = colors.read().unwrap().clone();
        //let read_pixel_data: Vec<PixelData> = pixel_data.read().unwrap().clone();

        source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % source.width()) as u16;
                let y = (i as u32 / source.width()) as u16;
                let mut p = Pixel::new(x, y, 0);
                let h = p.calc_heuristic(
                    (x, y),
                    target_pixels[i],
                    weights[i],
                    &read_colors,
                    settings.proximity_importance,
                    // &read_pixel_data,
                ) + STROKE_REWARD;
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>()
    };

    let mut rng = rand::thread_rng();
    fn max_dist(age: u32) -> u32 {
        (((DRAWING_CANVAS_SIZE / 4) as f32) * (0.99f32).powi(age as i32 / 30)).round() as u32
    }

    loop {
        let colors: Vec<SeedColor> = {
            let r = colors.read().unwrap();
            r.clone()
        };
        let pixel_data = {
            let r = pixel_data.read().unwrap();
            r.clone()
        };
        let mut swaps_made = 0;

        for _ in 0..SWAPS_PER_GENERATION {
            let apos = rng.gen_range(0..pixels.len());
            let ax = apos as u16 % target.width() as u16;
            let ay = apos as u16 / target.width() as u16;

            //let stroke_id = pixel_data[apos].stroke_id as usize;
            let max_dist_a = max_dist(frame_count.saturating_sub(pixel_data[apos].last_edited));

            let bx = (ax as i16 + rng.gen_range(-(max_dist_a as i16)..=(max_dist_a as i16)))
                .clamp(0, target.width() as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist_a as i16)..=(max_dist_a as i16)))
                .clamp(0, target.width() as i16 - 1) as u16;
            let bpos = by as usize * target.width() as usize + bx as usize;

            let max_dist_b = max_dist(frame_count.saturating_sub(pixel_data[bpos].last_edited));
            if (bx as i32 - ax as i32).abs() > max_dist_b as i32
                || (by as i32 - ay as i32).abs() > max_dist_b as i32
            {
                continue;
            }

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let a_on_b_h = pixels[apos].calc_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                &colors,
                settings.proximity_importance,
            ) + stroke_reward(bpos, apos, &pixel_data, &pixels, frame_count);

            let b_on_a_h = pixels[bpos].calc_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                &colors,
                settings.proximity_importance,
            ) + stroke_reward(apos, bpos, &pixel_data, &pixels, frame_count);

            let improvement_a = pixels[apos].h - b_on_a_h;
            let improvement_b = pixels[bpos].h - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        //println!("swaps made: {}", swaps_made);

        // let img = make_new_img(&source_pixels, &assignments, target.width());
        // if swaps_made < 10 || cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        //     let dir_name = save_result(target, base_name, source, assignments, img)?;
        //     tx.send(ProgressMsg::Done(PathBuf::from(format!(
        //         "./presets/{}",
        //         dir_name
        //     ))))?;
        //     return Ok(());
        // }
        // tx.send(ProgressMsg::UpdatePreview(img))?;
        if swaps_made > 0 {
            let assignments = pixels
                .iter()
                .map(|p| p.src_y as usize * source.width() as usize + p.src_x as usize)
                .collect::<Vec<_>>();
            tx.send(ProgressMsg::UpdateAssignments(assignments))?;
        }
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled).unwrap();
            return Ok(());
        }

        //max_dist = (max_dist as f32 * 0.99).max(4.0) as u32;
    }
}
fn load_weights(source: &image::DynamicImage) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}

fn serialize_assignments(assignments: Vec<usize>) -> String {
    format!(
        "[{}]",
        assignments
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}
