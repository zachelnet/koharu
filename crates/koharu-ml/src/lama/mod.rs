//! LaMa inference with IOPaint-compatible orchestration.

mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

pub use self::config::{HDStrategy, InpaintRequest};
use self::{config::FFCResNetGeneratorConfig, model::Model, processor::InpaintModel};

model_repository!("mayocream/lama-manga" @ "f91c85b26913b3e83f9877867b4c336da3675238" {
    WEIGHTS = "lama-manga.safetensors"
});

#[derive(Debug)]
pub struct LaMa {
    model: Model,
    processor: InpaintModel,
}

impl LaMa {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve LaMa weights")?;
        let mut model = Model::new(&FFCResNetGeneratorConfig::default(), device);
        model
            .load(&weights_path)
            .context("failed to load LaMa safetensors")?;
        Ok(Self {
            model,
            processor: InpaintModel::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| self.processor.call(&self.model, image, mask, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use koharu_runtime::{Feature, Runtime};

    async fn preload_libtorch() {
        Runtime::discover([Feature::Torch])
            .unwrap()
            .initialize()
            .await
            .unwrap();
    }

    /// Light background with a dark "text" bubble that LaMa must fill with the
    /// surrounding light color. A correct inpaint keeps the bubble light;
    /// backend/precision faults show up as a dark fill.
    fn synthetic_input() -> (RgbImage, GrayImage) {
        let mut image = RgbImage::from_pixel(256, 256, Rgb([235, 235, 235]));
        let mut mask = GrayImage::new(256, 256);
        for y in 80..176 {
            for x in 80..176 {
                image.put_pixel(x, y, Rgb([30, 30, 30]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        (image, mask)
    }

    fn mean_brightness(rgb: &RgbImage, mask: &GrayImage) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for (x, y, pixel) in rgb.enumerate_pixels() {
            if mask.get_pixel(x, y)[0] > 127 {
                sum += f64::from(pixel[0]) * 0.299
                    + f64::from(pixel[1]) * 0.587
                    + f64::from(pixel[2]) * 0.114;
                count += 1.0;
            }
        }
        sum / count
    }

    #[tokio::test]
    async fn rocm_vs_cpu_fill_quality() {
        preload_libtorch().await;
        let (image, mask) = synthetic_input();
        let input = DynamicImage::ImageRgb8(image);
        let request = InpaintRequest::default();

        // The accelerated device may be CUDA, ROCm, or a CPU fallback depending
        // on the host; compare against CPU rather than asserting an absolute
        // value. A corrupt reduced-precision ROCm kernel would darken the fill
        // by dozens of luminance units and fail the tolerance.
        let accelerated = LaMa::load(crate::Device::rocm(0))
            .await
            .expect("load accelerated LaMa");
        let accelerated_out = accelerated
            .inference(&input, &mask, &request)
            .expect("accelerated inference");
        let accelerated_mean = mean_brightness(&accelerated_out, &mask);

        let cpu = LaMa::load(crate::Device::cpu())
            .await
            .expect("load CPU LaMa");
        let cpu_out = cpu.inference(&input, &mask, &request).expect("CPU inference");
        let cpu_mean = mean_brightness(&cpu_out, &mask);

        println!(
            "inpainted-region mean brightness: accelerated={accelerated_mean:.1} cpu={cpu_mean:.1}"
        );
        assert!(
            (accelerated_mean - cpu_mean).abs() < 8.0,
            "accelerated LaMa fill diverges from CPU: {accelerated_mean:.1} vs {cpu_mean:.1}"
        );
    }
}
