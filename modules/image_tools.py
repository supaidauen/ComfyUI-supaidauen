from pathlib import Path
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
MAX_RESOLUTION = nodes.MAX_RESOLUTION

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
  RETURN_TYPES = ("IMAGE", "MASK"
                  , "IMAGE", "MASK"
                  , "IMAGE", "MASK"
                  , "IMAGE", "MASK"
                  , "IMAGE", "MASK"
                  ,"COMPOSITE_PIPE")
  RETURN_NAMES = ('image','mask'
                  ,'image1','mask1'
                  ,'image2','mask2'
                  ,'image3','mask3'
                  ,'image4','mask4'
                  ,'composite_pipe')
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
              image, mask,
              image1, mask1,
              image2, mask2,
              image3, mask3,
              image4, mask4,
              composite_pipe
          )
      if 'r' not in locals():
        r = pairs[i]
      r = self.composite(r,pairs[i])
    image,mask = r[0],r[1]
    composite_pipe = (image1,mask1,image2,mask2,image3,mask3,image4,mask4)
    return(image,mask,
          image1, mask1,
          image2, mask2,
          image3, mask3,
          image4, mask4,
          composite_pipe)

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

class supaidauen_Image_From_List:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
          "image_list": ("IMAGE", ),
          "image_index": ("INT", {"default":0, "step":1,}),
          }
    }
  
  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"

  def doit(self, image_list, image_index):
    return(image_list[image_index])