//! Rendering loop. Pixels → null geodesics → sky samples → PNG.
//!
//! Parallelism: rayon over pixels. Each ray is fully independent, so this
//! scales linearly with cores. On the B200 box we'll replace this with one
//! thread per CUDA core; the geodesic kernel is unchanged.

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::prelude::*;

use crate::camera::Camera;
use crate::geodesic::{mom, pos, trace};
use crate::metric::{Metric, Termination};
use crate::sky;
use crate::types::Scalar;

#[derive(Debug, Clone, Copy)]
pub struct RenderConfig {
    pub step: Scalar,
    pub max_steps: usize,
    /// Background colour for `Captured` rays (the hole).
    pub captured_rgb: [f32; 3],
    /// Background colour for `Diverged` rays (numerical failure). Magenta so
    /// it screams. If you see this in the image, the integrator failed —
    /// don't silently accept it.
    pub diverged_rgb: [f32; 3],
    /// √(rays per pixel). 1 = no supersampling; 2 = 4×; 3 = 9×; 4 = 16×.
    /// We use a regular grid rather than Halton/Sobol because the bottleneck
    /// is the geodesic integration, not the sample pattern.
    pub samples_per_axis: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            step: 0.05,
            max_steps: 20_000,
            captured_rgb: [0.0, 0.0, 0.0],
            diverged_rgb: [1.0, 0.0, 1.0],
            samples_per_axis: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub captured: usize,
    pub escaped: usize,
    pub diverged: usize,
    pub max_null_residual: Scalar,
    pub avg_steps: f64,
}

/// Render an image. Returns the linear RGB buffer (row-major, top-left
/// origin) and per-ray statistics for sanity checking.
pub fn render<M: Metric>(
    metric: &M,
    camera: &Camera,
    cfg: &RenderConfig,
) -> (Vec<f32>, RenderStats) {
    let tetrad = camera.tetrad(metric);
    let w = camera.width;
    let h = camera.height;
    let n_pixels = w * h;

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({percent}%) {msg}",
    )
    .unwrap();

    let spa = cfg.samples_per_axis.max(1);
    let inv_spa = 1.0 / spa as Scalar;

    let outcomes: Vec<([f32; 3], Termination, usize, Scalar)> = (0..n_pixels)
        .into_par_iter()
        .progress_with_style(style)
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;

            // Accumulate over sub-pixel samples; classify the pixel by the
            // dominant termination so the diagnostic stats stay meaningful.
            let mut rgb_acc = [0.0f32; 3];
            let mut total_steps_pix: usize = 0;
            let mut max_resid_pix: Scalar = 0.0;
            let mut nc = 0u32;
            let mut ne = 0u32;
            let mut nd = 0u32;

            for sj in 0..spa {
                for si in 0..spa {
                    let du = (si as Scalar + 0.5) * inv_spa;
                    let dv = (sj as Scalar + 0.5) * inv_spa;
                    let y0 = camera.ray_state_sub(&tetrad, i, j, du, dv);
                    let res = trace(metric, y0, cfg.step, cfg.max_steps);

                    let rgb = match res.termination {
                        Termination::Captured => {
                            nc += 1;
                            cfg.captured_rgb
                        }
                        Termination::Diverged => {
                            nd += 1;
                            cfg.diverged_rgb
                        }
                        Termination::Escaped => {
                            ne += 1;
                            let x = pos(&res.final_state);
                            let p = mom(&res.final_state);
                            let dir = metric.asymptotic_direction(&x, &p);
                            sky::sample(dir)
                        }
                    };
                    rgb_acc[0] += rgb[0];
                    rgb_acc[1] += rgb[1];
                    rgb_acc[2] += rgb[2];
                    total_steps_pix += res.steps_taken;
                    if res.null_residual.is_finite() && res.null_residual > max_resid_pix {
                        max_resid_pix = res.null_residual;
                    }
                }
            }

            let n = (spa * spa) as f32;
            let avg_rgb = [rgb_acc[0] / n, rgb_acc[1] / n, rgb_acc[2] / n];

            // Pick a representative classification: whichever outcome happened
            // most often. Ties go captured > diverged > escaped so divergence
            // is never hidden by a single bad sample.
            let term = if nd > 0 && nd >= nc && nd >= ne {
                Termination::Diverged
            } else if nc >= ne {
                Termination::Captured
            } else {
                Termination::Escaped
            };

            (
                avg_rgb,
                term,
                total_steps_pix / (spa * spa),
                max_resid_pix,
            )
        })
        .collect();

    let mut pixels = vec![0.0f32; n_pixels * 3];
    let mut stats = RenderStats::default();
    let mut total_steps: usize = 0;
    for (idx, (rgb, term, steps, resid)) in outcomes.iter().enumerate() {
        pixels[idx * 3] = rgb[0];
        pixels[idx * 3 + 1] = rgb[1];
        pixels[idx * 3 + 2] = rgb[2];
        match term {
            Termination::Captured => stats.captured += 1,
            Termination::Escaped => stats.escaped += 1,
            Termination::Diverged => stats.diverged += 1,
        }
        total_steps += steps;
        if *resid > stats.max_null_residual {
            stats.max_null_residual = *resid;
        }
    }
    stats.avg_steps = total_steps as f64 / n_pixels as f64;
    (pixels, stats)
}

/// Tonemap and write the buffer to a PNG. Linear → sRGB is a single gamma
/// step (we're not chasing physical brightness — the sky is procedural).
pub fn write_png(
    path: &std::path::Path,
    pixels: &[f32],
    width: usize,
    height: usize,
) -> image::ImageResult<()> {
    let gamma_inv = 1.0 / 2.2;
    let mut buf: Vec<u8> = Vec::with_capacity(pixels.len());
    for c in pixels {
        let v = c.clamp(0.0, 1.0).powf(gamma_inv);
        buf.push((v * 255.0).round() as u8);
    }
    let img = image::RgbImage::from_raw(width as u32, height as u32, buf)
        .expect("buffer size matches dimensions");
    img.save(path)
}
