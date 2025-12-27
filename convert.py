from safetensors.torch import load_file, save_file
import safetensors.torch
import torch

import gc

out_dtype = torch.float8_e4m3fn
if out_dtype == torch.float8_e4m3fn:
    bits = 8
    mantissa_bit = 3
    sign_bits = 1
    dtype_string = "e4m3fn"
elif out_dtype == torch.float8_e5m2:
    bits = 8
    mantissa_bit = 2
    sign_bits = 1
    dtype_string = "e5m2"
    
def fp8_tensor_quant(x, scale):
    scale = scale.reshape([-1] + [1] * (x.dim() - 1))
    quant_dequant_x = quantize_to_fp8(x / scale)
    return quant_dequant_x, scale

def quantize_to_fp8(x):
    device = x.device
    dtype = x.dtype
    M = max(1, min(mantissa_bit, bits - sign_bits))
    E = bits - sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = torch.tensor(mantissa * 2 ** (2**E - 1 - bias), device=device, dtype=dtype)
    minval = -maxval if sign_bits == 1 else torch.tensor(0, device=device, dtype=dtype)
    input_clamp = torch.clamp(x, minval.item(), maxval.item())
    eps = 1e-6
    log_scales = torch.floor(torch.log2(torch.abs(input_clamp) + eps) + bias)
    log_scales = torch.clamp_min(log_scales, 1.0)
    log_scales = 2.0 ** (log_scales - M - bias)
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out


def load_file(path):
    print(f"Loading {path}...")
    if not path.endswith(".safetensors"):
        loaded = torch.load(path)
    else:
        loaded = safetensors.torch.load_file(path)
    return loaded


sd_pruned = dict()
params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "text_embedding", "time_", "img_emb", "modulation"}
model_type = "Wan2_2-T2V-A14B-HIGH"

for i in range(1, 7):
    file_path = f"diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    current_sd = load_file(file_path)

    for k, v in current_sd.items():
        if k not in sd_pruned:
            if any(keyword in k for keyword in params_to_keep):
                if "patch_embedding" in k:
                    sd_pruned[k] = v
                else:
                    sd_pruned[k] = v.to(torch.float32)
            elif "weight" in k:
                if out_dtype == torch.float8_e4m3fn:
                    scale = torch.max(torch.abs(v))
                elif out_dtype == torch.float8_e5m2:
                    abs_v = torch.abs(v.cpu().flatten())
                    if abs_v.numel() > 1000000:
                        sample = abs_v[torch.randint(0, abs_v.numel(), (1000000,))]
                        scale = torch.quantile(sample, 0.999)
                    else:
                        scale = torch.quantile(abs_v, 0.999)
                qdq_out, scale_tensor = fp8_tensor_quant(v.cuda(), scale.cuda())
                print(f"scale_tensor for key {k}: {scale_tensor}")
                sd_pruned[k] = qdq_out.to(out_dtype)
                sd_pruned[k.replace(".weight", ".scale_weight")] = scale_tensor[0]
            else:
                sd_pruned[k] = v#.to(torch.float16)

    # Clear memory
    del current_sd
    gc.collect()
    torch.cuda.empty_cache()

sd_pruned["scaled_fp8"] = torch.tensor([0., 0.], dtype=out_dtype)

for k, v in sd_pruned.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape} {v.dtype}")
    else:
        print(f"{k}: {type(v)}")
save_file(sd_pruned, f"{model_type}_fp8_{dtype_string}_scaled_KJ.safetensors", metadata={"format": "pt", "model_type": model_type})
