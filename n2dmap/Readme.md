A small wrapper around the [`fft2d`] crate to convert normal maps in depth maps.

It is directly derived from the `fft2d` [example][fft2d-n2d].

It is a library that exposes a single function taking an `Image` in the format
of a normal map (a three channel image with 8 bit color components in linear
color space) and returning a depth map, with values between 0 and 1, where
0 is the closest to the surface, while 1 is the deepest.

[`fft2d`]: https://github.com/mpizenberg/fft2d
[fft2d-n2d]: https://github.com/mpizenberg/fft2d/blob/main/examples/normal_integration.rs
