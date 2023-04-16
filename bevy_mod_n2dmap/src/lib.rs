use bevy::prelude::{Assets, Image, StandardMaterial};
use image::DynamicImage;
use n2dmap::normal_to_depth;

pub trait N2dmapMaterialExtension {
    /// Generate depth map based on normal map using [`n2dmap`].
    fn generate_depth_map(&mut self, images: &mut Assets<Image>) -> bool;
}

impl N2dmapMaterialExtension for StandardMaterial {
    fn generate_depth_map(&mut self, images: &mut Assets<Image>) -> bool {
        let Some(normal_map) = &self.normal_map_texture else { return false; };
        let Some(normal_map) = images.get(normal_map) else { return false; };
        let Ok(normal_map) = normal_map.clone().try_into_dynamic() else { return false; };
        let Ok(depth_map) = normal_to_depth(&normal_map.into_rgb8()) else { return false; };
        let depth_map = DynamicImage::ImageLuma8(depth_map);
        let depth_map = images.add(Image::from_dynamic(depth_map, false));
        self.depth_map = Some(depth_map);
        true
    }
}
