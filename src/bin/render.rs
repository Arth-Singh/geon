//! Render binary. `cargo run --release -- --metric schwarzschild --width 1024
//! --height 1024 --out outputs/schwarzschild.png`.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use geon::metric::alcubierre::Alcubierre;
use geon::metric::minkowski::Minkowski;
use geon::metric::morris_thorne::MorrisThorne;
use geon::metric::schwarzschild::Schwarzschild;
use geon::{Camera, RenderConfig, render, write_png};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MetricKind {
    Minkowski,
    Schwarzschild,
    Alcubierre,
    MorrisThorne,
}

#[derive(Parser, Debug)]
#[command(version, about = "Render a black hole / wormhole / warp bubble.")]
struct Args {
    #[arg(long, value_enum, default_value_t = MetricKind::Schwarzschild)]
    metric: MetricKind,

    #[arg(long, default_value_t = 1024)]
    width: usize,

    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// Field of view, degrees, vertical.
    #[arg(long, default_value_t = 60.0)]
    fov: f64,

    /// Observer radial coordinate (units of M for Schwarzschild).
    #[arg(long, default_value_t = 20.0)]
    r_obs: f64,

    /// Integrator step size (affine parameter).
    #[arg(long, default_value_t = 0.05)]
    step: f64,

    /// Maximum integration steps per ray.
    #[arg(long, default_value_t = 20_000)]
    max_steps: usize,

    /// √(rays per pixel). 1=none, 2=4×SS, 3=9×, 4=16×.
    #[arg(long, default_value_t = 1)]
    samples: usize,

    /// Disable the procedural starfield. Useful for proving that "speckle in
    /// the shadow" is really a coherent secondary-image ring, not noise.
    #[arg(long)]
    no_stars: bool,

    /// Output PNG path.
    #[arg(long, default_value = "outputs/render.png")]
    out: PathBuf,

    /// Alcubierre bubble velocity (in units of c). 0.5 = subluminal, well-behaved.
    /// 2.0 = superluminal, fun but the integrator struggles in the walls.
    #[arg(long, default_value_t = 0.5)]
    bubble_v: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.no_stars {
        geon::sky::SHOW_STARS.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    let cfg = RenderConfig {
        step: args.step,
        max_steps: args.max_steps,
        samples_per_axis: args.samples,
        ..Default::default()
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!(
        "rendering {:?} @ {}x{}, fov {}°, r_obs {}, step {}",
        args.metric, args.width, args.height, args.fov, args.r_obs, args.step
    );

    let stats = match args.metric {
        MetricKind::Minkowski => {
            let metric = Minkowski;
            // Minkowski Cartesian chart: observer at (0, 20, 0, 0).
            let cam = Camera::new([0.0, args.r_obs, 0.0, 0.0], args.fov, args.width, args.height);
            let (pixels, stats) = render(&metric, &cam, &cfg);
            write_png(&args.out, &pixels, args.width, args.height)?;
            stats
        }
        MetricKind::Schwarzschild => {
            let metric = Schwarzschild::new(1.0);
            let cam = Camera::new(
                [0.0, args.r_obs, std::f64::consts::FRAC_PI_2, 0.0],
                args.fov,
                args.width,
                args.height,
            );
            let (pixels, stats) = render(&metric, &cam, &cfg);
            write_png(&args.out, &pixels, args.width, args.height)?;
            stats
        }
        MetricKind::Alcubierre => {
            let metric = Alcubierre {
                ship_velocity: args.bubble_v,
                ..Alcubierre::default()
            };
            let cam = Camera::new([0.0, args.r_obs, 0.0, 0.0], args.fov, args.width, args.height);
            let (pixels, stats) = render(&metric, &cam, &cfg);
            write_png(&args.out, &pixels, args.width, args.height)?;
            stats
        }
        MetricKind::MorrisThorne => {
            let metric = MorrisThorne::default();
            // Observer at l = r_obs (one of the two universes), looking at throat.
            let cam = Camera::new(
                [0.0, args.r_obs, std::f64::consts::FRAC_PI_2, 0.0],
                args.fov,
                args.width,
                args.height,
            );
            let (pixels, stats) = render(&metric, &cam, &cfg);
            write_png(&args.out, &pixels, args.width, args.height)?;
            stats
        }
    };

    println!(
        "done. captured={}, escaped={}, diverged={}, avg_steps={:.1}, max_null_residual={:.3e}",
        stats.captured, stats.escaped, stats.diverged, stats.avg_steps, stats.max_null_residual,
    );
    println!("wrote {}", args.out.display());

    Ok(())
}
