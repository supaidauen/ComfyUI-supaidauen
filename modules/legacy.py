class supaidauen_Recursive_Uspcaler:
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": { 
          "upscale_model": ("UPSCALE_MODEL",),
          "image": ("IMAGE",),
          "iterations": ("INT", {"default": 1}),
          }
      }
  RETURN_TYPES = ("IMAGE", )
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, upscale_model, image, iterations, ):
    images = ImageUpscaleWithModel().upscale(upscale_model, image)[0]
    if iterations == 1:
      return(images, )
    for i in range(1,iterations):
      images = ImageUpscaleWithModel().upscale(upscale_model, images)[0]
    return(images, )

class Subject_Detection_and_Interrupt:
  def __init__(self):
    pass
  #
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image_list": ("IMAGE",),
        "image_passthrough": ("IMAGE",),
        "min": ("INT", {"default": 1}), 
        "max": ("INT", {"default": 1})
      }
    }
  #
  RETURN_TYPES = ("INT","IMAGE",)
  RETURN_NAMES = ("Count","Image")
  INPUT_IS_LIST = True
  OUTPUT_IS_LIST = (False, False, )
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, image_list, image_passthrough, min=1, max=1):
    count = int(ImageBatchToCount().doit(image_list)[0])
    if not min[0]<=count<=max[0]:
      nodes.interrupt_processing(True)
    image = image_passthrough[0]
    return (count, image)

class ClearVRAM:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "empty_cache": ("BOOLEAN", {"default": True}),
        "gc_collect": ("BOOLEAN", {"default": True}),
        "unload_all_models": ("BOOLEAN", {"default": False}),
      },
      "optional": {
        "image_pass": ("IMAGE",),
        "model_pass": ("MODEL",),
      }
    }
  #
  RETURN_TYPES = ("IMAGE","MODEL","INT", "INT",)
  RETURN_NAMES = ("image_pass", "model_pass", "freemem_before", "freemem_after")
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  DESCRIPTION = """
    Returns the inputs unchanged, they are only used as triggers,  
    and performs comfy model management functions and garbage collection,  
    reports free VRAM before and after the operations.
    """
  
  def doit(self, gc_collect,empty_cache, unload_all_models, image_pass=None, model_pass=None):
    freemem_before = model_management.get_free_memory()
    print("VRAMdebug: free memory before: ", freemem_before)
    if empty_cache:
      model_management.soft_empty_cache()
    if unload_all_models:
      model_management.unload_all_models()
    if gc_collect:
      import gc
      gc.collect()
    freemem_after = model_management.get_free_memory()
    print("VRAMdebug: free memory after: ", freemem_after)
    print("VRAMdebug: freed memory: ", freemem_after - freemem_before)
    return (image_pass, model_pass, freemem_before, freemem_after)

class ImageBatchToCount:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",), 
      }
    }
  #
  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("Count",)
  INPUT_IS_LIST = True
  OUTPUT_IS_LIST = (False,)
  FUNCTION = "doit"
  #
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, images):
    count = len(images)
    return (count, )

class supaidauen_Normalized_Float_Slider:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
      },
    }

  RETURN_TYPES = ("FLOAT", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, value):
    return (float(value), )
