use std::{fs, mem, path::PathBuf};

use crate::{SeedColor, SeedPos};
use image::GenericImageView;

pub fn init_image(sidelen: u32, source_dir: PathBuf) -> (u32, Vec<SeedPos>, Vec<SeedColor>, Sim) {
    // load rust_output/source.png
    let source = image::open(source_dir.join("source.png")).unwrap();
    let assignments = fs::read_to_string(source_dir.join("assignments.json"))
        .unwrap()
        .strip_prefix('[')
        .unwrap()
        .strip_suffix(']')
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect::<Vec<usize>>();

    let mut seeds = Vec::new();
    let mut colors = Vec::new();
    let mut sim = Sim::new(source_dir);

    let width = source.width() as usize;
    let height = source.height() as usize;

    assert_eq!(width, height);

    let seeds_n = width * height;
    let pixelsize = sidelen as f32 / width as f32;

    for y in 0..width {
        for x in 0..width {
            let p = source.get_pixel(x as u32, y as u32);
            seeds.push(SeedPos {
                xy: [(x as f32 + 0.5) * pixelsize, (y as f32 + 0.5) * pixelsize],
            });
            colors.push(SeedColor {
                rgba: [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    1.0,
                ],
            });
        }
    }
    sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.0); seeds_n];

    for (dst_idx, src_idx) in assignments.iter().enumerate() {
        let src_x = (src_idx % width) as f32;
        let src_y = (src_idx / width) as f32;
        let dst_x = (dst_idx % width) as f32;
        let dst_y = (dst_idx / width) as f32;

        let dst_force = 0.13;

        sim.cells[*src_idx] = CellBody::new(
            (src_x + 0.5) * pixelsize,
            (src_y + 0.5) * pixelsize,
            (dst_x + 0.5) * pixelsize,
            (dst_y + 0.5) * pixelsize,
            dst_force,
        );
    }

    (seeds_n as u32, seeds, colors, sim)
}

#[derive(Clone, Copy)]
struct CellBody {
    srcx: f32,
    srcy: f32,
    dstx: f32,
    dsty: f32,

    velx: f32,
    vely: f32,

    accx: f32,
    accy: f32,

    dst_force: f32,
}

const PERSONAL_SPACE: f32 = 0.95;
const MAX_VELOCITY: f32 = 6.0;
const ALIGNMENT_FACTOR: f32 = 0.7;

fn factor_curve(x: f32) -> f32 {
    if x < 10.0 {
        x * x * x
    } else {
        1000.0
    }
}

impl CellBody {
    fn new(srcx: f32, srcy: f32, dstx: f32, dsty: f32, dst_force: f32) -> Self {
        Self {
            srcx,
            srcy,
            dstx,
            dsty,
            dst_force,
            velx: 0.0,
            vely: 0.0,
            accx: 0.0,
            accy: 0.0,
        }
    }

    fn update(&mut self, pos: &mut SeedPos) {
        self.velx += self.accx;
        self.vely += self.accy;

        self.accx = 0.0;
        self.accy = 0.0;

        self.velx *= 0.97;
        self.vely *= 0.97;

        // self.velx = self.velx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        // self.vely = self.vely.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        // pos.xy[0] += self.velx;
        // pos.xy[1] += self.vely;

        pos.xy[0] += self.velx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        pos.xy[1] += self.vely.clamp(-MAX_VELOCITY, MAX_VELOCITY);
    }

    fn apply_dst_force(&mut self, pos: &SeedPos, sidelen: f32, elapsed: f32) {
        let factor = factor_curve(elapsed * self.dst_force);

        let dx = self.dstx - pos.xy[0];
        let dy = self.dsty - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();

        self.accx += (dx * dist * factor) / sidelen;
        self.accy += (dy * dist * factor) / sidelen;
    }

    fn apply_neighbour_force(&mut self, pos: &SeedPos, other: &SeedPos, pixel_size: f32) -> f32 {
        let dx = other.xy[0] - pos.xy[0];
        let dy = other.xy[1] - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();
        let personal_space = pixel_size * PERSONAL_SPACE;

        let weight = (1.0 / dist) * (personal_space - dist) / personal_space;

        if dist > 0.0 && dist < personal_space {
            self.accx -= dx * weight;
            self.accy -= dy * weight;
        } else if dist.abs() < f32::EPSILON {
            // if they are exactly on top of each other, push in a random direction
            self.accx += (rand::random::<f32>() - 0.5) * 0.1;
            self.accy += (rand::random::<f32>() - 0.5) * 0.1;
        }

        weight.max(0.0)
    }

    fn apply_wall_force(&mut self, pos: &SeedPos, sidelen: f32, pixel_size: f32) {
        let personal_space = pixel_size * PERSONAL_SPACE * 0.5;

        if pos.xy[0] < personal_space {
            self.accx += (personal_space - pos.xy[0]) / personal_space;
        } else if pos.xy[0] > sidelen - personal_space {
            self.accx -= (pos.xy[0] - (sidelen - personal_space)) / personal_space;
        }

        if pos.xy[1] < personal_space {
            self.accy += (personal_space - pos.xy[1]) / personal_space;
        } else if pos.xy[1] > sidelen - personal_space {
            self.accy -= (pos.xy[1] - (sidelen - personal_space)) / personal_space;
        }
    }
}

pub struct Sim {
    elapsed_frames: u32,
    cells: Vec<CellBody>,
    source: PathBuf,
}

impl Sim {
    pub fn new(source_dir: PathBuf) -> Self {
        Self {
            cells: Vec::new(),
            elapsed_frames: 0,
            source: source_dir,
        }
    }

    pub fn name(&self) -> String {
        preset_path_to_name(&self.source)
    }

    pub fn source_path(&self) -> PathBuf {
        self.source.clone()
    }

    pub fn switch(&mut self) {
        for cell in &mut self.cells {
            mem::swap(&mut cell.srcx, &mut cell.dstx);
            mem::swap(&mut cell.srcy, &mut cell.dsty);
        }
        self.elapsed_frames = 0;
    }

    pub fn update(&mut self, positions: &mut Vec<SeedPos>, sidelen: u32) {
        let grid_size = (self.cells.len() as f32).sqrt();
        let pixel_size = sidelen as f32 / grid_size;
        //dbg!(grid_size, pixel_size);

        let mut grid = vec![vec![]; self.cells.len()];

        for (i, p) in positions.iter().enumerate() {
            let x = p.xy[0] / pixel_size;
            let y = p.xy[1] / pixel_size;

            let index = (y.floor().clamp(0.0, grid_size - 1.0) * grid_size) as usize
                + (x.floor().clamp(0.0, grid_size - 1.0) as usize);
            //
            grid[index].push(i);
        }

        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.apply_wall_force(&positions[i], sidelen as f32, pixel_size);
            cell.apply_dst_force(
                &positions[i],
                sidelen as f32,
                self.elapsed_frames as f32 / 60.0,
            );
        }

        for i in 0..self.cells.len() {
            let pos = positions[i].xy;
            let col = (pos[0] / pixel_size) as usize;
            let row = (pos[1] / pixel_size) as usize;
            let mut avg_xvel = 0.0;
            let mut avg_yvel = 0.0;
            let mut count = 0.0;
            for dy in 0..=2 {
                for dx in 0..=2 {
                    if col + dx == 0
                        || row + dy == 0
                        || col + dx >= grid_size as usize
                        || row + dy >= grid_size as usize
                    {
                        continue;
                    }
                    let ncol = col + dx - 1;
                    let nrow = row + dy - 1;
                    let nindex = nrow * (grid_size as usize) + ncol;
                    for other in grid[nindex].iter() {
                        if other == &i {
                            continue;
                        }
                        let other_cell = positions[*other];
                        let weight = self.cells[i].apply_neighbour_force(
                            &positions[i],
                            &other_cell,
                            pixel_size,
                        );

                        avg_xvel += self.cells[*other].velx * weight;
                        avg_yvel += self.cells[*other].vely * weight;
                        count += weight;
                    }
                }
            }

            if count > 0.0 {
                avg_xvel /= count;
                avg_yvel /= count;

                self.cells[i].accx += (avg_xvel - self.cells[i].velx) * ALIGNMENT_FACTOR;
                self.cells[i].accy += (avg_yvel - self.cells[i].vely) * ALIGNMENT_FACTOR;
            }
        }

        for (index, cell) in self.cells.iter_mut().enumerate() {
            cell.update(&mut positions[index]);
        }

        self.elapsed_frames += 1;
    }
}

pub fn preset_path_to_name(source_dir: &PathBuf) -> String {
    source_dir
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}
