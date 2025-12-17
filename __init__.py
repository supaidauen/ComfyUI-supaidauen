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

class supaidauen_Image_Compositor:
  MAX = 3
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(self):
    items = list(range(self.MAX + 1))
    options = []

    # combination sizes 1..MAX+1
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            # label shown in the dropdown
            label = "[" + ",".join(str(x) for x in combo) + "]"
            # actual returned value
            options.append((label, list(combo)))
    # ComfyUI dropdowns support (label, value) entries  
    labels = [label for label, value in options]
    values = {label: value for label, value in options}
    return {
      "required": {
        "selection": (labels, {"default": labels[0], "values": values}),
      },
      "optional": {
        "image1": ('IMAGE',),
        "mask1": ('MASK',),
        "image2": ('IMAGE',),
        "mask2": ('MASK',),
        "image3": ('IMAGE',),
        "mask3": ('MASK',),
        "image4": ('IMAGE',),
        "mask4": ('MASK',),
        "composite_pipe": ('COMPOSITE_PIPE',),
      }
    }
  #
  RETURN_TYPES = ("IMAGE", "MASK","MASK","MASK","MASK","MASK","COMPOSITE_PIPE")
  RETURN_NAMES = ('image','mask','mask1','mask2','mask3','mask4','composite_pipe')
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  @staticmethod
  def combine(self, destination, source, x, y, operation):
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1]))

    left, top = (x, y,)
    right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
    visible_width, visible_height = (right - left, bottom - top,)

    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = output[:, top:bottom, left:right]

    if operation == "multiply":
        output[:, top:bottom, left:right] = destination_portion * source_portion
    elif operation == "add":
        output[:, top:bottom, left:right] = destination_portion + source_portion
    elif operation == "subtract":
        output[:, top:bottom, left:right] = destination_portion - source_portion
    elif operation == "and":
        output[:, top:bottom, left:right] = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
    elif operation == "or":
        output[:, top:bottom, left:right] = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
    elif operation == "xor":
        output[:, top:bottom, left:right] = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()

    output = torch.clamp(output, 0.0, 1.0)

    return (output,)
  
  @staticmethod
  def composite(pair1=[], pair2=[]):
    source = pair1[0]
    destination = pair2[0]
    source_mask = pair1[1]
    destination_mask = pair2[1]
    if source == None or destination == None or source_mask == None or destination_mask == None:
       return (source, source_mask)
    output = destination_mask.reshape((-1, destination_mask.shape[-2], destination_mask.shape[-1])).clone()
    source_mask = source_mask.reshape((-1, source_mask.shape[-2], source_mask.shape[-1]))
    left, top = (0, 0,)
    right, bottom = (min(left + source_mask.shape[-1], destination_mask.shape[-1]), min(top + source_mask.shape[-2], destination_mask.shape[-2]))
    visible_width, visible_height = (right - left, bottom - top,)
    source_portion = source_mask[:, :visible_height, :visible_width]
    destination_portion = output[:, top:bottom, left:right]
    output[:, top:bottom, left:right] = 1-(1-(destination_portion) + 1-(source_portion))
    mask = torch.clamp(output, 0.0, 1.0)

    image, = ImageCompositeMasked().composite(destination, source, 0, 0, resize_source=False, mask=pair2[1])
    return (image, mask)

  FIELDS = (
    "image1", "mask1",
    "image2", "mask2",
    "image3", "mask3",
    "image4", "mask4",
  )
  @staticmethod
  def merge_composite_pipe(composite_pipe, **overrides):
    # Start with composite_pipe or empty defaults
    if composite_pipe is not None:
        base = dict(zip(supaidauen_Image_Compositor.FIELDS, composite_pipe))
    else:
        base = {k: None for k in supaidauen_Image_Compositor.FIELDS}
    # Override only explicitly-provided values
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    return tuple(base[k] for k in supaidauen_Image_Compositor.FIELDS)

  def doit(
    self,
    selection,
    image1=None, mask1=None,
    image2=None, mask2=None,
    image3=None, mask3=None,
    image4=None, mask4=None,
    composite_pipe=None,
  ):
    # Merge composite_pipe + overrides
    image1, mask1, image2, mask2, image3, mask3, image4, mask4 = \
        self.merge_composite_pipe(
            composite_pipe,
            image1=image1, mask1=mask1,
            image2=image2, mask2=mask2,
            image3=image3, mask3=mask3,
            image4=image4, mask4=mask4,
        )
  
    pairs = [(image1,mask1),(image2,mask2),(image3,mask3),(image4,mask4)]
    selected = json.loads(selection)
    for i in selected:
      if len(selected) == 1:
          image, mask = pairs[i]
          composite_pipe = (
              image1, mask1,
              image2, mask2,
              image3, mask3,
              image4, mask4
          )
          return (
              image,
              mask,
              mask1,
              mask2,
              mask3,
              mask4,
              composite_pipe
          )
      if 'r' not in locals():
        r = pairs[i]
      r = self.composite(r,pairs[i])
    image,mask = r[0],r[1]
    composite_pipe = (image1,mask1,image2,mask2,image3,mask3,image4,mask4)
    return(image,mask,mask1,mask2,mask3,mask4,composite_pipe)

class supaidauen_LoadImageFromPath_input:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("STRING", {"default": "", "forceInput": True}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
        }
    }
  #
  CATEGORY = "supaidauen/Util"
  #
  RETURN_TYPES = ("IMAGE", "MASK")
  FUNCTION = "load_image"
  #
  def load_image(self, image, **kwargs):
    if isinstance(image, list):
      image = image[0]
    image_path = folder_paths.get_annotated_filepath(image)
  #
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
      mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
      mask = 1. - torch.from_numpy(mask)
    else:
      mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image, mask)
  #
  @staticmethod
  def _resolve_path(image) -> Path:
      image_path = Path(folder_paths.get_annotated_filepath(image))
      return image_path
  #
  @classmethod
  def VALIDATE_INPUTS(s, image):
    # If image is an output of another node, it will be None during validation
    if image is None:
        return True
  #
    image_path = s._resolve_path(image)
    if not image_path.exists():
        return "Invalid image path: {}".format(image_path)
  #
    return True

class supaidauen_GenerateRandomImagePadding:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "pad_multiplier": ("INT", {"default": 8, "min": 0, "max": 64} ),
        "pad_range": ("INT", {"default": 0, "min": 0, "max": 8} ),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
        },
      "optional":{
        "extra_padding_left": ("INT", {"default": 0, "min": -8096, "max": MAX_RESOLUTION}),
        "extra_padding_right": ("INT", {"default": 0, "min": -8096, "max": MAX_RESOLUTION}),
        "extra_padding_top": ("INT", {"default": 0, "min": -8096, "max": MAX_RESOLUTION}),
        "extra_padding_bottom": ("INT", {"default": 0, "min": -8096, "max": MAX_RESOLUTION}),
        },
      "hidden": {
        "unique_id": "UNIQUE_ID",
      }
    }
  #
  CATEGORY = "supaidauen/Util"
  #
  RETURN_TYPES = ("INT", "INT", "INT", "INT")
  RETURN_NAMES = ("left","right","top","bottom")
  FUNCTION = "doit"
  #
  def doit(self, seed, pad_multiplier, pad_range, unique_id,extra_padding_left,extra_padding_right,extra_padding_top,extra_padding_bottom):
    paddings = [ pad * pad_multiplier for pad in range(0,pad_range+1)]
    sides = {
      "left":1,
      "right":2,
      "top":3,
      "bottom":4}
    
    def gen_padding(i):
      random.seed(seed+i)
      return(random.choice(paddings))
      

    for side in sides:
      sides[side] = gen_padding(sides[side]+int(unique_id))
    return (sides["left"]+extra_padding_left,
            sides["right"]+extra_padding_right,
            sides["top"]+extra_padding_top,
            sides["bottom"]+extra_padding_bottom)

class supaidauen_Character_IO:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "name": ("STRING", {"multiline": True, "default": "", "defaultInput": False}),
        "path": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "suffix": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "wildcard": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
      },
      "optional":{
        "width": ("INT", {"default": 1024, "min": -8096, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": -8096, "max": MAX_RESOLUTION}),
        "rescale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step":0.25, "round": 0.01}),
        "crop_to_size": ("BOOLEAN", {"default": False,"label_on": "resize","label_off": "default"}),
      }
    }
  #
  RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
  RETURN_NAMES = ("image", "mask", "width", "height")
  FUNCTION = "load_image"
  CATEGORY = "supaidauen/Util"
  #
  def load_image(self, name, path, suffix, wildcard, seed, width, height, rescale, crop_to_size,):
    image = self._get_filename(name, path, suffix, wildcard)
    #
    if not self._resolve_path(image):
        return "Invalid image path: {}".format(image_path)
    if isinstance(image, list):
      image = image[0]
    image_path = folder_paths.get_annotated_filepath(image)
    #
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    if crop_to_size:
      default_width = i.size[0]
      default_height = i.size[1]
      diff_width = default_width - width
      diff_height = default_height - height
      left = diff_width/2
      upper = diff_height/2
      right = width+left
      lower = height+upper
      new_size = (left, upper, right, lower)
      print(new_size)
      i = i.crop(new_size)
      i = i.resize([round(size*rescale) for size in i.size])
    img_width = i.size[0]
    img_height = i.size[1]
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
      mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
      mask = 1. - torch.from_numpy(mask)
    else:
      mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image, mask, img_width, img_height)
  #
  @staticmethod
  def _get_filename(name, path, suffix, wildcard):
    filename = f"{path}{suffix}".replace(wildcard,name)
    return filename
  #
  @staticmethod
  def _resolve_path(image) -> Path:
      image_path = Path(folder_paths.get_annotated_filepath(image))
      return image_path
  #

class supaidauen_ImagePadding:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
          "image": ("IMAGE", ),
          "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
          },
        "optional": {
        }
        }
  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("images",)
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"

  def doit(self, image, left, right, top, bottom,width,height):
    B, H, W, C = image.shape
    bg_color = [int(x)/255.0 for x in [0,0,0]] 
    bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)
    pad_left = left
    pad_right = right
    pad_top = top
    pad_bottom = bottom

    padded_width = W + pad_left + pad_right
    padded_height = H + pad_top + pad_bottom
    out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
    
    # Fill padded areas
    for b in range(B):
      # Pad with edge color
      # Define edge pixels
      top_edge = image[b, 0, :, :]
      bottom_edge = image[b, H-1, :, :]
      left_edge = image[b, :, 0, :]
      right_edge = image[b, :, W-1, :]

      # Fill borders with edge colors
      out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
      out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
      out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
      out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
      out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
    
    image = out_image

    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    multiple_of = 1
    width = width - (width % multiple_of)
    height = height - (height % multiple_of)

    width = width if width > 0 else ow
    height = height if height > 0 else oh

    outputs = image.permute(0,3,1,2)
    outputs = F.interpolate(outputs, size=(height, width), mode='nearest')
    outputs = outputs.permute(0,2,3,1)

    if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
      width = outputs.shape[2]
      height = outputs.shape[1]
      x = (width % multiple_of) // 2
      y = (height % multiple_of) // 2
      x2 = width - ((width % multiple_of) - x)
      y2 = height - ((height % multiple_of) - y)
      outputs = outputs[:, y:y2, x:x2, :]
    
    outputs = torch.clamp(outputs, 0, 1)

    return(outputs, outputs.shape[2], outputs.shape[1],)

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
}
