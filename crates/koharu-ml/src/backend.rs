use anyhow::{Result, bail};
use koharu_runtime::{Backend, Device, Hardware};
use koharu_torch::{Kind, nn};

static CPU_FALLBACK_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) trait TryIntoDevice<T> {
    fn try_into_device(self) -> Result<T>;
}

impl TryIntoDevice<koharu_torch::Device> for Device {
    fn try_into_device(self) -> Result<koharu_torch::Device> {
        match self.backend {
            Backend::Cpu => Ok(koharu_torch::Device::Cpu),
            Backend::Cuda | Backend::Rocm => {
                if koharu_torch::utils::has_cuda() || koharu_torch::utils::has_hip() {
                    Ok(koharu_torch::Device::Cuda(self.index))
                } else {
                    if !CPU_FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::warn!(
                            backend = %self.backend,
                            "Torch was built without CUDA/HIP; falling back to CPU"
                        );
                    }
                    Ok(koharu_torch::Device::Cpu)
                }
            }
            Backend::Vulkan if self.index == 0 => Ok(if koharu_torch::utils::has_vulkan() {
                koharu_torch::Device::Vulkan
            } else {
                koharu_torch::Device::Cpu
            }),
            Backend::Metal
                if self.index == 0 && cfg!(all(target_os = "macos", target_arch = "aarch64")) =>
            {
                Ok(koharu_torch::Device::Mps)
            }
            Backend::Vulkan | Backend::Metal => {
                bail!(
                    "Torch cannot address {} device index {}",
                    self.backend,
                    self.index
                )
            }
            Backend::Other(_) => bail!(
                "the {} backend cannot be represented by a Torch device",
                self.backend
            ),
        }
    }
}

pub(crate) fn set_precision(var_store: &mut nn::VarStore) {
    let device = var_store.device();
    let kind = if let koharu_torch::Device::Cuda(_) = device {
        // Discovery probes CUDA/HIP/Vulkan, so only run it when the model
        // actually runs on an accelerator.
        let hardware = Hardware::discover();
        // ROCm: FP16 overflows and corrupts output.
        if hardware.supports_rocm() {
            if hardware.rocm_supports_bf16() {
                Kind::BFloat16
            } else {
                Kind::Float
            }
        } else if hardware.cuda_compute_capability() >= 80 {
            Kind::BFloat16
        } else {
            Kind::Half
        }
    } else {
        Kind::Float
    };

    if var_store.is_empty() {
        var_store.set_kind(kind);
    } else if kind == Kind::BFloat16 {
        var_store.bfloat16();
    } else if kind == Kind::Half {
        var_store.half();
    } else {
        var_store.float();
    }
}
