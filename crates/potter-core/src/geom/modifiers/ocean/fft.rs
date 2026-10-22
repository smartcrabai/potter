// SPDX-License-Identifier: GPL-3.0-or-later
//! Unnormalized complex transforms for the ocean height field.
//!
//! Blender evaluates the ocean with FFTW `c2r` plans, which are unnormalized. The ocean output
//! scale depends on that unnormalized sum (the Jacobian and foam are not normalized by the height
//! factor), so this module never divides by the transform length.

use std::f64::consts::TAU;

/// Double-precision complex number used by the ocean transforms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Complex {
    pub(super) re: f64,
    pub(super) im: f64,
}

impl Complex {
    pub(super) fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub(super) fn from_polar(radius: f64, phase: f64) -> Self {
        Self {
            re: radius * phase.cos(),
            im: radius * phase.sin(),
        }
    }

    pub(super) fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub(super) fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Plan for an unnormalized inverse DFT of one length: `X[k] = sum_j x[j] exp(+2 pi i j k / n)`.
pub(super) struct InverseDft {
    len: usize,
    kind: Kind,
}

enum Kind {
    Radix2(Radix2),
    /// Chirp-z evaluation of an arbitrary length on a power-of-two convolution.
    Bluestein {
        /// `exp(+i pi j^2 / len)` for `j < len`.
        chirp: Vec<Complex>,
        /// Forward FFT of the conjugate chirp, zero padded to the convolution length.
        kernel: Vec<Complex>,
        convolution: Box<Radix2>,
    },
}

/// Power-of-two transform used inside Bluestein and for radix-2 lengths.
struct Radix2 {
    len: usize,
    /// `exp(-2 pi i k / len)` for `k < len / 2`.
    forward: Vec<Complex>,
    /// `exp(+2 pi i k / len)` for `k < len / 2`.
    inverse: Vec<Complex>,
}

impl Radix2 {
    fn new(len: usize) -> Self {
        let half = len / 2;
        let mut forward = Vec::with_capacity(half);
        let mut inverse = Vec::with_capacity(half);
        for k in 0..half {
            let phase = TAU * k as f64 / len as f64;
            forward.push(Complex::from_polar(1.0, -phase));
            inverse.push(Complex::from_polar(1.0, phase));
        }
        Self {
            len,
            forward,
            inverse,
        }
    }

    fn transform(&self, values: &mut [Complex], inverse: bool) {
        let len = self.len;
        let mut j = 0;
        for i in 1..len {
            let mut bit = len >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                values.swap(i, j);
            }
        }
        let table = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        let mut block = 2;
        while block <= len {
            let half = block / 2;
            let stride = len / block;
            for start in (0..len).step_by(block) {
                for offset in 0..half {
                    let twiddle = table[offset * stride];
                    let even = values[start + offset];
                    let odd = values[start + offset + half] * twiddle;
                    values[start + offset] = even + odd;
                    values[start + offset + half] = even - odd;
                }
            }
            block <<= 1;
        }
    }
}

impl InverseDft {
    pub(super) fn new(len: usize) -> Self {
        if len.is_power_of_two() {
            return Self {
                len,
                kind: Kind::Radix2(Radix2::new(len)),
            };
        }
        let convolution_len = (2 * len - 1).next_power_of_two();
        let convolution = Radix2::new(convolution_len);
        let chirp: Vec<Complex> = (0..len)
            .map(|j| {
                // j^2 modulo 2 len keeps the phase argument small for large lengths.
                let square = (j as u128 * j as u128) % (2 * len as u128);
                Complex::from_polar(1.0, std::f64::consts::PI * square as f64 / len as f64)
            })
            .collect();
        let mut kernel = vec![Complex::default(); convolution_len];
        for (j, value) in kernel.iter_mut().enumerate().take(len) {
            *value = chirp[j].conj();
        }
        for j in 1..len {
            kernel[convolution_len - j] = chirp[j].conj();
        }
        convolution.transform(&mut kernel, false);
        Self {
            len,
            kind: Kind::Bluestein {
                chirp,
                kernel,
                convolution: Box::new(convolution),
            },
        }
    }

    /// Transforms `values` in place. `values.len()` must equal the planned length.
    pub(super) fn inverse(&self, values: &mut [Complex]) {
        debug_assert_eq!(values.len(), self.len);
        match &self.kind {
            Kind::Radix2(plan) => plan.transform(values, true),
            Kind::Bluestein {
                chirp,
                kernel,
                convolution,
            } => {
                let convolution_len = convolution.len;
                let mut padded = vec![Complex::default(); convolution_len];
                for (j, value) in values.iter().enumerate() {
                    padded[j] = *value * chirp[j];
                }
                convolution.transform(&mut padded, false);
                for (value, kernel_value) in padded.iter_mut().zip(kernel) {
                    *value = *value * *kernel_value;
                }
                convolution.transform(&mut padded, true);
                let scale = 1.0 / convolution_len as f64;
                for (k, value) in values.iter_mut().enumerate() {
                    *value = padded[k].scale(scale) * chirp[k];
                }
            }
        }
    }
}

/// Unnormalized 2D `c2r` transform of a Hermitian-half spectrum, the semantics of FFTW's
/// `fftw_plan_dft_c2r_2d` used by Blender.
///
/// `half` is `rows x (cols / 2 + 1)` in row-major order, `half[i * (cols / 2 + 1) + j]`. Columns are
/// transformed over the complete row index first, then each row is completed by Hermitian mirroring
/// and its real part is kept, which is what the FFTW output contains for non-Hermitian input too.
pub(super) struct C2r2d {
    rows: usize,
    cols: usize,
    row_plan: InverseDft,
    col_plan: InverseDft,
}

impl C2r2d {
    pub(super) fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_plan: InverseDft::new(cols),
            col_plan: InverseDft::new(rows),
        }
    }

    /// Returns the real `rows x cols` output in row-major order.
    pub(super) fn run(&self, half: &[Complex]) -> Vec<f64> {
        let rows = self.rows;
        let cols = self.cols;
        let half_cols = cols / 2 + 1;
        debug_assert_eq!(half.len(), rows * half_cols);
        let mut transformed = vec![Complex::default(); rows * half_cols];
        let mut column = vec![Complex::default(); rows];
        for j in 0..half_cols {
            for (i, value) in column.iter_mut().enumerate() {
                *value = half[i * half_cols + j];
            }
            self.col_plan.inverse(&mut column);
            for (x, value) in column.iter().enumerate() {
                transformed[x * half_cols + j] = *value;
            }
        }
        let mut output = vec![0.0; rows * cols];
        let mut row = vec![Complex::default(); cols];
        for x in 0..rows {
            for (j, value) in row.iter_mut().enumerate() {
                *value = if j < half_cols {
                    transformed[x * half_cols + j]
                } else {
                    transformed[x * half_cols + (cols - j)].conj()
                };
            }
            self.row_plan.inverse(&mut row);
            for (y, value) in row.iter().enumerate() {
                output[x * cols + y] = value.re;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::{C2r2d, Complex, InverseDft};

    fn naive_inverse(values: &[Complex]) -> Vec<Complex> {
        let len = values.len();
        (0..len)
            .map(|k| {
                values
                    .iter()
                    .enumerate()
                    .fold(Complex::default(), |sum, (j, value)| {
                        let phase = std::f64::consts::TAU * (j * k) as f64 / len as f64;
                        sum + *value * Complex::from_polar(1.0, phase)
                    })
            })
            .collect()
    }

    fn assert_close(left: &[Complex], right: &[Complex]) {
        for (a, b) in left.iter().zip(right) {
            assert!((a.re - b.re).abs() < 1.0e-9 && (a.im - b.im).abs() < 1.0e-9);
        }
    }

    #[test]
    fn inverse_dft_matches_direct_sum_for_power_and_arbitrary_lengths() {
        for len in [1, 2, 3, 5, 8, 9, 16, 25, 49] {
            let input: Vec<Complex> = (0..len)
                .map(|j| Complex::new((j as f64 * 0.37).sin(), (j as f64 * 1.3).cos()))
                .collect();
            let mut actual = input.clone();
            InverseDft::new(len).inverse(&mut actual);
            assert_close(&actual, &naive_inverse(&input));
        }
    }

    #[test]
    fn c2r_equals_real_part_of_mirrored_full_inverse_sum() {
        for (rows, cols) in [(3, 4), (5, 5), (4, 6)] {
            let half_cols = cols / 2 + 1;
            let half: Vec<Complex> = (0..rows * half_cols)
                .map(|index| Complex::new((index as f64 * 0.7).sin(), (index as f64 * 0.3).cos()))
                .collect();
            let output = C2r2d::new(rows, cols).run(&half);
            for x in 0..rows {
                for y in 0..cols {
                    let mut sum = Complex::default();
                    for i in 0..rows {
                        for j in 0..cols {
                            let value = if j < half_cols {
                                half[i * half_cols + j]
                            } else {
                                half[((rows - i) % rows) * half_cols + (cols - j)].conj()
                            };
                            let phase = std::f64::consts::TAU
                                * ((i * x) as f64 / rows as f64 + (j * y) as f64 / cols as f64);
                            sum = sum + value * Complex::from_polar(1.0, phase);
                        }
                    }
                    assert!((output[x * cols + y] - sum.re).abs() < 1.0e-9);
                }
            }
        }
    }
}
