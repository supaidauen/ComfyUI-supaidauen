from pathlib import Path
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.nodes_mask import ImageCompositeMasked

import folder_paths
import nodes

import random
import itertools
import json
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None

# Modules Imports
from .modules.passthroughs import *
from .modules.legacy import *
from .modules.text import *
from .modules.impact_overrides import *
from .modules.conditioning_per_image import *
from .modules.universal_tiled_sampler import *
from .modules.random_prompt_text import *
from .modules.image_tools import *

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class supaidauen_Integer:
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required":{
          "integer": ("INT", {"default": 0}),
          }
      }
  RETURN_TYPES = ('INT',)
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, integer, ):
    return(integer,)

class KSampler_Advanced_Calculator:
  def __init__(self):
      pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "starting_step": ("INT", {"default": 1}),
        "steps_offset": ("INT", {"default": 10}),
        "percent_additional_steps": ("FLOAT", {"default": 0, "min": 0, "step":0.01, "round": 0.01}),
        "cfg": ("FLOAT", {"default": 1, "min": 0, "step":0.01, "round": 0.01}),
        "number_additional_steps": ("INT", {"default": 0, "min": 0}),
        "percent_or_number": ("BOOLEAN", {"default": True,"label_on": "percent","label_off": "number"}),
      },
      "optional": {
      },
    }
  #
  RETURN_TYPES = ("INT","INT","INT", "FLOAT","FLOAT")
  RETURN_NAMES = ("steps", "start_at_step", "end_at_step", "cfg","denoise")
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  DESCRIPTION = """
    Calculates the amount of steps you would like the KSampler (Advanced)
    to take after starting from an initial step. Also provides the CFG.
    """
  #
  def doit(self, starting_step, steps_offset, percent_additional_steps, number_additional_steps, cfg,percent_or_number,):
    start_at_step = starting_step
    end_at_step = starting_step+steps_offset
    if percent_or_number:
      steps = (int(end_at_step*(1+percent_additional_steps)))
      print(percent_or_number)
    else:
      steps = (int(end_at_step+number_additional_steps))
      print(percent_or_number)
    get_denoise = (steps - starting_step) / steps
    return {"result": (steps, start_at_step, end_at_step, cfg, get_denoise)}

class Latent_Switcher:
  def __init__(self):
      pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "upscale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step":0.05, "round": 0.01}),
        "upscale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step":0.05, "round": 0.01}),
        "upscale_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step":0.05, "round": 0.01}),
        "portrait_latent": ("LATENT",),
        "lanscape_latent": ("LATENT",),
        "latent_toggle": ("BOOLEAN", {"default": True,"label_on": "landscape","label_off": "portrait"}),
      },
      "optional": {
      }
    }
  #
  RETURN_TYPES = ("FLOAT","FLOAT","LATENT",)
  RETURN_NAMES = ("upscale_15", "upscale_XL", "latent",)
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  DESCRIPTION = """
    Allows toggling of portait or landscape latents and sets some 
    neural net latent upscaler parameters based on the selection
    """
  #
  def doit(self, upscale_1,upscale_2,upscale_3,portrait_latent,lanscape_latent,latent_toggle):
    if latent_toggle == True:
      upscale_XL = upscale_1
      upscale_15 = upscale_3
      latent = lanscape_latent
    else:
      upscale_XL = upscale_3
      upscale_15 = upscale_2
      latent = portrait_latent
    return (upscale_15, upscale_XL, latent,)

class supaidauen_DummyRandomInt :
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "min" : ("INT", {"default": 0, "forceInput": False}),
        "max" : ("INT", {"default": 0, "forceInput": False}),
        "seed" : ("INT", {"default": 0, "forceInput": True}),
      },
      "optional": {},
    }
  #
  RETURN_TYPES = ("INT", )
  RETURN_NAMES = ("integer", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, min=0, max=0, seed=0):
      random.seed(seed)
      integer = random.randint(min,max)
      return (integer, )

NODE_CLASS_MAPPINGS = {
  "Supaidauen_LoadImageFromPath_input": supaidauen_LoadImageFromPath_input,
  "Integer": supaidauen_Integer,
  "Supaidauen_Recursive_Uspcaler": supaidauen_Recursive_Uspcaler,
  "ImageBatchToCount": ImageBatchToCount,
  "Subject_Detection_and_Interrupt": Subject_Detection_and_Interrupt,
  "ClearVRAM": ClearVRAM,
  "Supaidauen_Prompt_Consolidator": supaidauen_PromptConsolidator,
  "Supaidauen_Text_Concat": supaidauen_TextConcatenate,
  "Supaidauen_Text_w_Options_Replace_LoRA": supaidauen_Text_w_Options_Replace_LoRA,
  "Supaidauen_Create_Filename": supaidauen_TextConcatenateFileName,
  "Supaidauen_Text_Replace": supaidauen_TextReplace,
  "Supaidauen_Text_Wildcard": supaidauen_TextWildcard,
  "Supaidauen_Add_RunID": supaidauen_RunIDConcatenate,
  "Supaidauen_Normalized_Float_Slider": supaidauen_Normalized_Float_Slider,
  "Supaidauen_Create_DummyRandomInt": supaidauen_DummyRandomInt,
  "Supaidauen_GenerateRandomImagePadding": supaidauen_GenerateRandomImagePadding,
  "Supaidauen_Character_IO": supaidauen_Character_IO,
  "Supaidauen_ImagePadding": supaidauen_ImagePadding,
  "Supaidauen_Image_Compositor": supaidauen_Image_Compositor,
  "Supaidauen_Passthrough_VAE": supaidauen_passthrough_VAE,
  "Supaidauen_Passthrough_IMAGE": supaidauen_passthrough_IMAGE,
  "Supaidauen_Passthrough_LATENT": supaidauen_passthrough_LATENT,
  "Supaidauen_Passthrough_CLIP": supaidauen_passthrough_CLIP,
  "Supaidauen_Passthrough_MASK": supaidauen_passthrough_MASK,
  "Supaidauen_Passthrough_STRING": supaidauen_passthrough_STRING,
  "Supaidauen_ImpactEndAtStepModelControl": EndAtStepsModelControl,
  "Supaidauen_ZippedPromptFromTextAdvanced": supaidauen_ZippedPromptFromTextAdvanced,
  "Supaidauen_KREA_Masker": supaidauen_KREA_Masker,
  "UniversalTiledSampler": UniversalTiledSamplerNode,
  "RandomPromptText": RandomPromptText,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
  "Supaidauen_LoadImageFromPath_input": "Image Load using input",
  "Integer":"Integer",
  "ImageBatchToCount": "Image Batch To Count",
  "Subject_Detection_and_Interrupt": "Subject Detection and Interrupt",
  "ClearVRAM": "Clear the VRAM",
  "KSampler_Advanced_Calculator": "KSampler (Advanced) Calculator",
  "Latent_Switcher": "Latent Switcher",
  "Supaidauen_Prompt_Consolidator": "Simple Prompt Consolidator",
  "Supaidauen_Text_Concat": "Simple Text Concatenation",
  "Supaidauen_Text_Replace": "Simple Text with Replace",
  "Supaidauen_Text_Wildcard": "Simple Text Wildcard Replacement",
  "Supaidauen_Text_w_Options_Replace_LoRA": "Modify Wildcards",
  "Supaidauen_Create_Filename": "Parametric Filename",
  "Supaidauen_Add_RunID": "Add Run ID",
  "Supaidauen_Create_DummyRandomInt": "Dummy Random Integer",
  "Supaidauen_Image_Compositor": "Composite Images",
  "Supaidauen_Normalized_Float_Slider": "Normalized Float Slider",
  "Supaidauen_GenerateRandomImagePadding": "Generate Random Image Padding",
  "Supaidauen_Character_IO": "Load Character Images",
  "Supaidauen_ImagePadding": "Pad the Image",
  "Supaidauen_Passthrough_VAE": "Passthrough VAE",
  "Supaidauen_Passthrough_IMAGE": "Passthrough IMAGE",
  "Supaidauen_Passthrough_LATENT": "Passthrough LATENT",
  "Supaidauen_Passthrough_CLIP": "Passthrough CLIP",
  "Supaidauen_Passthrough_MASK": "Passthrough MASK",
  "Supaidauen_Passthrough_STRING": "Passthrough STRING",
  "Supaidauen_ImpactEndAtStepModelControl": "Detailer For Each Pipe (End At Step)",
  "Supaidauen_ZippedPromptFromTextAdvanced": "Zipped prompt from string",
  "Supaidauen_KREA_Masker": "KREA Rebalance Widget",
  "UniversalTiledSampler": "Universal Tiled Sampler",
  "RandomPromptText": "Random Prompt Text (Seeded)",
}
