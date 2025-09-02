use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use egui::ahash::AHasher;
use image::GenericImageView;
use pathfinding::prelude::Weights;
use std::sync::mpsc;

pub struct GenerationSettings {
    proximity_importance: i64,
    rescale: Option<u32>,
}

impl GenerationSettings {
    pub fn default() -> Self {
        Self {
            proximity_importance: 15,
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

    fn at(&self, row: usize, col: usize) -> i64 {
        let (x1, y1) = (row % self.sidelen, row / self.sidelen);
        let (x2, y2) = (col % self.sidelen, col / self.sidelen);
        let dist = (x1 as i64 - x2 as i64).pow(2) + (y1 as i64 - y2 as i64).pow(2);

        let (r1, g1, b1) = self.target[row];
        let (r2, g2, b2) = self.source[col];

        let dr = r1 as i64 - r2 as i64;
        let dg = g1 as i64 - g2 as i64;
        let db = b1 as i64 - b2 as i64;

        let out = -((dr.pow(2) + dg.pow(2) + db.pow(2)) * self.weights[row]
            + (dist * self.settings.proximity_importance).pow(2));

        out
    }

    fn neg(&self) -> Self {
        todo!()
    }
}

pub enum ProgressMsg {
    Progress(f32),
    Done(PathBuf), // result directory
    Error(String),
    Cancelled,
}

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<AHasher>>;

pub fn process<P: AsRef<Path>>(
    source_path: P,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let start_time = std::time::Instant::now();
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
    let base_name = source_path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let source = image::open(source_path)?.to_rgb8();
    // rescale source to match target dimensions
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

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target.pixels().map(|p| (p[0], p[1], p[2])).collect(),
        weights: load_weights(&target_weights.into()),
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

    let mut img = image::ImageBuffer::new(target.width(), target.height());

    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (x, y) = (
            (target_idx % target.width() as usize) as u32,
            (target_idx / target.width() as usize) as u32,
        );
        let (r, g, b) = source_pixels[*source_idx];
        img.put_pixel(x, y, image::Rgb([r, g, b]));
    }

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
