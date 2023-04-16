// SPDX-License-Identifier: MPL-2.0

use std::error::Error;

use fft2d::nalgebra::dcst::{dct_2d, idct_2d};
use image::{ImageBuffer, Luma, Primitive, Rgb};
use nalgebra::{DMatrix, Scalar, Vector2, Vector3};

pub type NormalMap = ImageBuffer<Rgb<u8>, Vec<u8>>;
pub type DepthMap = ImageBuffer<Luma<u8>, Vec<u8>>;

pub fn normal_to_depth(normal_map: NormalMap) -> Result<DepthMap, Box<dyn Error>> {
    // Extract normals.
    let mut normals = matrix_from_rgb_image(&normal_map, |x| *x as f32 / 255.0);
    for n in normals.iter_mut() {
        if n.x + n.y + n.z != 0.0 {
            n.x = (n.x - 0.5) * 2.0;
            n.y = (n.y - 0.5) * -2.0;
            n.z = (n.z - 0.5).max(0.001) * -2.0;
            n.normalize_mut();
        }
    }
    let depths = normal_integration(&normals);

    // Visualize depths.
    // TODO: clamp outsider values.
    // Note that the stannum impl takes the 1/4 3/4 min/max rather than full
    // min/max, then clamps
    let depth_min = depths.min();
    let depth_max = depths.max();
    eprintln!("depths within [ {},  {} ]", depth_min, depth_max);
    // let depths_to_gray =
    //     |z| ((z - depth_min) / (depth_max - depth_min) * (256.0 * 256.0 - 1.0)) as u16;
    let depth_to_u8 = |z: &f32| {
        let scaled = (z - depth_min) / (depth_max - depth_min);
        (scaled * 256.0) as u8
    };
    Ok(image_from_matrix(&depths, depth_to_u8))
}

/// Orthographic integration of a normal field into a depth map.
/// WARNING: normals is transposed since comming from an RGB image.
fn normal_integration(normals: &DMatrix<Vector3<f32>>) -> DMatrix<f32> {
    let (nrows, ncols) = normals.shape();

    // Compute gradient of the log depth map.
    let mut gradient_x = DMatrix::zeros(nrows, ncols);
    let mut gradient_y = DMatrix::zeros(nrows, ncols);
    for ((gx, gy), n) in gradient_x
        .iter_mut()
        .zip(gradient_y.iter_mut())
        // normals is 3 x npixels
        .zip(normals.iter())
    {
        // Only assign gradients different than 0
        // for pixels where the slope isn't too steep.
        if n.z < -0.01 {
            *gx = -n.x / n.z;
            *gy = -n.y / n.z;
        }
    }

    // Depth map by Poisson solver, up to an additive constant.
    dct_poisson(&gradient_y, &gradient_x)
}

/// An implementation of a Poisson equation solver with a DCT
/// (integration with Neumann boundary condition).
///
/// This code is based on the description in Section 3.4 of the paper:
/// [1] Normal Integration: a Survey - Queau et al., 2017
///
/// ```rust
/// u = dct_poisson(p, q);
/// ```
///
/// Where `p` and `q` are MxN matrices, solves in the least square sense
///
/// $$\nabla u = [ p, q ]$$
///
/// assuming the natural Neumann boundary condition on boundaries.
///
/// $$\nabla u \cdot \eta = [ p , q ] \cdot \eta$$
///
/// Remarq: for this solver, the "x" axis is considered to be
/// in the direction of the first coordinate of the matrix.
///
/// ```
/// Axis: O --> y
///       |
///       v
///       x
/// ```
fn dct_poisson(p: &DMatrix<f32>, q: &DMatrix<f32>) -> DMatrix<f32> {
    let (nrows, ncols) = p.shape();

    // Compute the divergence f = px + qy
    // of (p,q) using central differences (right-hand side of Eq. 30 in [1]).

    // Compute divergence term of p for the center of the matrix.
    let mut px = DMatrix::zeros(nrows, ncols);
    let p_top = p.slice_range(0..nrows - 2, ..);
    let p_bottom = p.slice_range(2..nrows, ..);
    px.slice_range_mut(1..nrows - 1, ..)
        .copy_from(&(0.5 * (p_bottom - p_top)));

    // Special treatment for the first and last rows (Eq. 52 in [1]).
    px.row_mut(0).copy_from(&(0.5 * (p.row(1) - p.row(0))));
    px.row_mut(nrows - 1)
        .copy_from(&(0.5 * (p.row(nrows - 1) - p.row(nrows - 2))));

    // Compute divergence term of q for the center of the matrix.
    let mut qy = DMatrix::zeros(nrows, ncols);
    let q_left = q.slice_range(.., 0..ncols - 2);
    let q_right = q.slice_range(.., 2..ncols);
    qy.slice_range_mut(.., 1..ncols - 1)
        .copy_from(&(0.5 * (q_right - q_left)));

    // Special treatment for the first and last columns (Eq. 52 in [1]).
    qy.column_mut(0)
        .copy_from(&(0.5 * (q.column(1) - q.column(0))));
    qy.column_mut(ncols - 1)
        .copy_from(&(0.5 * (q.column(ncols - 1) - q.column(ncols - 2))));

    // Divergence.
    let mut f = px + qy;

    // Natural Neumann boundary condition.
    let mut f_left = f.column_mut(0);
    f_left += q.column(0);
    let mut f_top = f.row_mut(0);
    f_top += p.row(0);
    let mut f_bottom = f.row_mut(nrows - 1);
    f_bottom -= p.row(nrows - 1);
    let mut f_right = f.column_mut(ncols - 1);
    f_right -= q.column(ncols - 1);

    // Cosine transform of f.
    // WARNING: a transposition occurs here with the dct2d.
    let mut f_cos = dct_2d(f.map(|x| x as f64));

    // Cosine transform of z (Eq. 55 in [1])
    let pi = std::f64::consts::PI;
    let coords = coordinates_column_major((ncols, nrows));
    for (f_cos_ij, (i, j)) in f_cos.iter_mut().zip(coords) {
        let v = Vector2::new(i as f64 / ncols as f64, j as f64 / nrows as f64);
        let denom = 4.0 * v.map(|u| (0.5 * pi * u).sin()).norm_squared();
        *f_cos_ij /= -(denom.max(1e-7));
    }

    // Inverse cosine transform:
    let depths = idct_2d(f_cos);

    // Z is known up to a positive constant, so offset it to get from 0 to max.
    // Also apply a normalization for the dct2d and idct2d.
    let z_min = depths.min();
    let dct_norm_coef = 4.0 / (nrows * ncols) as f32;
    depths.map(|z| dct_norm_coef * (z - z_min) as f32)
}

/// Iterator of the shape (row, column) where row increases first.
fn coordinates_column_major(shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let (nrows, ncols) = shape;
    (0..ncols).flat_map(move |j| (0..nrows).map(move |i| (i, j)))
}

// Image <-> Matrix ######################################################################

/// Convert a matrix into a gray level image.
/// Inverse operation of `matrix_from_image`.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
fn image_from_matrix<'a, T, F, U>(mat: &'a DMatrix<T>, to_gray: F) -> ImageBuffer<Luma<U>, Vec<U>>
where
    U: 'static + Primitive,
    T: Scalar,
    F: Fn(&'a T) -> U,
{
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([to_gray(&mat[(y as usize, x as usize)])]);
    }
    img_buf
}

/// Convert an RGB image into a `Vector3<T>` RGB matrix.
/// Inverse operation of `rgb_from_matrix`.
fn matrix_from_rgb_image<'a, T, F, U>(
    img: &'a ImageBuffer<Rgb<T>, Vec<T>>,
    scale: F,
) -> DMatrix<Vector3<U>>
where
    T: 'static + Primitive,
    U: Scalar,
    F: Fn(&'a T) -> U,
{
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (width, height) = img.dimensions();
    DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw()
            .chunks_exact(3)
            .map(|s| Vector3::new(scale(&s[0]), scale(&s[1]), scale(&s[2]))),
    )
    .transpose()
}
