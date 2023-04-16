# n2dmap

Crates to handle conversion from **normal** map (n) **to depth** map (d)
in bevy and outside of bevy.

This crate is very much just a wrapper around the `fft2d` [normal integration
example][fft2d-n2d], with additional improvements, mostly to make it work
nicely with bevy.

### [`n2dmap`]

A small wrapper around the [`fft2d`] crate to convert normal maps in depth maps.

It is directly derived from the `fft2d` [example][fft2d-n2d].

It is a library that exposes a single function taking an `Image` in the format
of a normal map (a three channel image with 8 bit color components in linear
color space) and returning a depth map, with values between 0 and 1, where
0 is the closest to the surface, while 1 is the deepest.

### [`n2dmap_cli`]

It's just an executable binary version of `n2dmap`.
See `n2dmap --help` for usage details.

### [`bevy_mod_n2dmap`]

A plugin for [bevy], exposes the `N2dmapMaterialExtension` trait, which adds the
`generate_depth_map()` method to bevy's `StandardMaterial`.

### License

All crates are licensed under the [MPL-2.0], as the original code is.

This is not legal advice, please read the MPL-2.0 license for details:
* You are free to re-use this code in your closed source software, as long as
  all changes to **files in this repository** are redistributed under the same
  licensing terms. You also need to include the license file in any distribution
  of software using MPL 2.0 code.
* In any case, conversion should happen offline, so you probably don't want to
  distribute this code 🤷

[bevy]: bevyengine.org/
[MPL-2.0]: https://spdx.org/licenses/MPL-2.0.html
[`n2dmap`]: ./n2dmap
[`bevy_mod_n2dmap`]: ./bevy_mod_n2dmap
[`n2dmap_cli`]: ./n2dmap_cli
[`fft2d`]: https://github.com/mpizenberg/fft2d
[fft2d-n2d]: https://github.com/mpizenberg/fft2d/blob/main/examples/normal_integration.rs