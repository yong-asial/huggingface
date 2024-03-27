# pip install flash_attn

import torch

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format

API_TOKEN = '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'
DEVICE = torch.device("cuda")

PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=API_TOKEN,
)
MODEL = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=API_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

