//! `geon` — differentiable, GPU-ready general-relativity engine.
//!
//! Named after J.A. Wheeler's 1955 *geon* — a self-gravitating bundle of
//! fields. The library is about the substance of spacetime: parameterise a
//! metric, integrate geodesics through it, compute its curvature, and
//! (Phase 3) gradient-descend through the space of metrics under causal
//! and energy-condition objectives.
//!
//! Phase 1 (current): CPU geodesic ray tracer. Schwarzschild + scaffolding
//! for Alcubierre / Morris–Thorne.
//! Phase 2: render the exotic metrics, quantify their energy-condition
//! violations numerically.
//! Phase 3: parameterise g_μν as a neural field, gradient-descend through
//! the space of spacetimes under causal-structure objectives.
//!
//! Convention: Lorentzian signature (-, +, +, +). Geometric units G = c = 1.

pub mod camera;
pub mod curvature;
pub mod geodesic;
pub mod integrator;
pub mod metric;
pub mod render;
pub mod sky;
pub mod types;

pub use camera::Camera;
pub use metric::{Metric, Termination};
pub use render::{RenderConfig, RenderStats, render, write_png};
pub use types::{Mat4, Scalar, Vec4};
