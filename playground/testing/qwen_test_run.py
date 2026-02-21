import json
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="cuda"
).eval()

# IMPORTANT: avoid double-resize because process_vision_info already handles vision preprocessing
processor = AutoProcessor.from_pretrained(MODEL_ID, do_resize=False, use_fast=False)
patch_size = int(processor.image_processor.patch_size)
merge_size = int(processor.image_processor.merge_size)

image_path = "/home/klrshak/work/VisionLang/whereami-text2sgm/out_dir/0ad2d382-79e2-2212-98b3-641bf9d552c1_topdown.png"
scene_description = "A bright white floor and matching white wall with pictures dominate the view, the scene looks like a living room. A white sofa is visible in front of the wall, and have blue pillows on it and chairs on either side."

# Read original size (what YOU care about for pixel predictions)
img0 = Image.open(image_path).convert("RGB")
W0, H0 = img0.size

prompt_text = f"""
You are a vision model helping with top-down localization and heading estimation.

Return ONLY a JSON object with keys u,v,w:
{{"u": <float>, "v": <float>, "w": <float>}}

Coordinate convention:
- Origin (0,0) is top-left of the image.
- u increases to the right, v increases downward.
- u must be in [0, {W0}), v must be in [0, {H0}).

Angle convention:
- w in degrees in [0,360)
- w = 0° points right (+u)
- w increases CCW (90° up, 180° left, 270° down)

IMPORTANT:
- u,v must refer to pixel coordinates of the ORIGINAL input image size {W0}x{H0}.

Scene description:
<<<
{scene_description}
>>>
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},  # keep full image, no manual crop
            {"type": "text", "text": prompt_text},
        ],
    }
]

# Qwen vision preprocessing (may resize; usually not crop)
image_inputs, video_inputs = process_vision_info(messages, image_patch_size=patch_size)

if image_inputs:
    image_seq = image_inputs if isinstance(image_inputs, (list, tuple)) else [image_inputs]
    required_multiple = patch_size * merge_size
    for idx, image in enumerate(image_seq):
        seen_w, seen_h = image.size
        if (seen_w % required_multiple) != 0 or (seen_h % required_multiple) != 0:
            raise ValueError(
                "Vision preprocessing mismatch: image "
                f"index {idx} resized to {seen_w}x{seen_h}, which is not divisible by "
                f"patch_size*merge_size={patch_size}*{merge_size}={required_multiple}. "
                "Check qwen_vl_utils image_patch_size and model processor settings."
            )

# Track what size the model actually saw (for mapping if needed)
seen = image_inputs[0] if isinstance(image_inputs, (list, tuple)) else image_inputs
W1, H1 = getattr(seen, "size", (None, None))  # PIL gives (W,H)

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
).to(model.device)

gen_cfg = GenerationConfig(
    max_new_tokens=128,
    do_sample=False,
    # temperature=0.0,
)

with torch.no_grad():
    out = model.generate(**inputs, generation_config=gen_cfg)

gen = out[0][inputs["input_ids"].shape[-1]:]
response = processor.decode(gen, skip_special_tokens=True).strip()
print("RAW:", response)

# Extract JSON robustly (handles ```json ... ``` fences)
start = response.find("{")
end = response.rfind("}")
if start == -1 or end == -1 or end <= start:
    raise ValueError("No JSON object found in model output")

pred = json.loads(response[start:end + 1])

u = pred["u"]
v = pred["v"]
w = pred["w"]

for name, value in (("u", u), ("v", v), ("w", w)):
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {name} in model output, got {type(value).__name__}")

# If the model implicitly predicted in "seen" coords, map to original.
# (This is a safety net; prompt asks for original coords, but models sometimes ignore.)
if W1 and H1 and (W1 != W0 or H1 != H0):
    # assume uniform resize without crop/pad
    u = float(u) * (W0 / W1)
    v = float(v) * (H0 / H1)

# Pixel outputs should be integer coordinates.
u = int(round(u))
v = int(round(v))

# Clamp + normalize only if needed; if any value is out of bounds, normalize all.
if not (0 <= u < W0 and 0 <= v < H0 and 0 <= w < 360):
    u = max(0, min(u, W0 - 1))
    v = max(0, min(v, H0 - 1))
    w = w % 360

print("PARSED:", {"u": u, "v": v, "w": w})
