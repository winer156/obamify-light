use std::path::{Path, PathBuf};

use egui::ahash::AHasher;
use image::GenericImageView;
use pathfinding::{kuhn_munkres::kuhn_munkres, prelude::Weights};
use std::sync::mpsc;

struct ImgDiffWeights {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
}

const PROXIMITY_IMPORTANCE: i64 = 12; // turn into user controlled option?
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

        let out = -((dr.pow(2) + dg.pow(2) + db.pow(2) * self.weights[row])
            + (dist * PROXIMITY_IMPORTANCE).pow(2));

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
}

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<AHasher>>;

pub fn process<P: AsRef<Path>>(source_path: P, tx: mpsc::SyncSender<ProgressMsg>) {
    let target = image::open(TARGET_IMAGE_PATH).unwrap().to_rgb8();
    let weights = image::open(TARGET_WEIGHTS_PATH).unwrap().to_rgb8();
    if target.dimensions().0 != target.dimensions().1 {
        tx.send(ProgressMsg::Error("Target image must be square".into()))
            .unwrap();
        return;
    }
    if target.dimensions() != weights.dimensions() {
        tx.send(ProgressMsg::Error(
            "Target and weights images must have the same dimensions".into(),
        ))
        .unwrap();
        return;
    }
    let base_name = source_path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let source = image::open(source_path).unwrap().to_rgb8();
    // rescale source to match target dimensions
    let source = image::imageops::resize(
        &source,
        target.width(),
        target.height(),
        image::imageops::FilterType::Triangle,
    );

    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target.pixels().map(|p| (p[0], p[1], p[2])).collect(),
        weights: load_weights(&image::open(TARGET_WEIGHTS_PATH).unwrap()),
        sidelen: target.width() as usize,
    };

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

            tx.send(ProgressMsg::Progress(root as f32 / nx as f32))
                .unwrap();
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

    std::fs::create_dir_all(format!("./presets/{}", dir_name)).unwrap();
    img.save(format!("./presets/{}/output.png", dir_name))
        .unwrap();
    source
        .save(format!("./presets/{}/source.png", dir_name))
        .unwrap();
    target
        .save(format!("./presets/{}/target.png", dir_name))
        .unwrap();
    std::fs::write(
        format!("./presets/{}/assignments.json", dir_name),
        serialize_assignments(assignments),
    )
    .unwrap();

    tx.send(ProgressMsg::Done(PathBuf::from(format!(
        "./presets/{}",
        dir_name
    ))))
    .unwrap();
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
