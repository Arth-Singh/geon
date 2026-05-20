//! Flat Minkowski spacetime. Sanity-check metric — geodesics here must be
//! straight lines in our coordinates, and the rendered image must show the
//! sky texture undistorted. If this fails, the whole framework is broken.

use super::{Metric, Termination};
use crate::types::{Mat4, Vec4, ZERO_MAT4};

pub struct Minkowski;

impl Metric for Minkowski {
    fn g(&self, _x: &Vec4) -> Mat4 {
        let mut m = ZERO_MAT4;
        m[0][0] = -1.0;
        m[1][1] = 1.0;
        m[2][2] = 1.0;
        m[3][3] = 1.0;
        m
    }

    fn dg(&self, _x: &Vec4) -> [Mat4; 4] {
        [ZERO_MAT4; 4]
    }

    fn g_inv(&self, x: &Vec4) -> Mat4 {
        self.g(x)
    }

    fn christoffel(&self, _x: &Vec4) -> crate::types::Christoffel {
        [[[0.0; 4]; 4]; 4]
    }

    fn terminate(&self, x: &Vec4) -> Option<Termination> {
        let r2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        if r2 > 1e6 {
            Some(Termination::Escaped)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "minkowski"
    }
}
