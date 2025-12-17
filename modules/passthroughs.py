class supaidauen_passthrough_VAE:

  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
      "vae": ('VAE',)}
    }
  #
  @classmethod
  def IS_CHANGED(s, vae):
    return True
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("VAE", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, vae):
    return(vae,)

class supaidauen_passthrough_IMAGE:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
      "image": ('IMAGE',)}
    }
  #
  @classmethod
  def IS_CHANGED(s, image):
    return True
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("IMAGE", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, image):
    return(image,)

class supaidauen_passthrough_LATENT:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
      "latent": ('LATENT',)}
    }
  #
  @classmethod
  def IS_CHANGED(s, latent):
    return True
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("LATENT", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, latent):
    return(latent,)

class supaidauen_passthrough_CLIP:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "clip": ('CLIP',)
      }
    }
  #
  @classmethod
  def IS_CHANGED(s, clip):
    return True
  #
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("CLIP", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, clip):
    return(clip,)

class supaidauen_passthrough_CLIP:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "clip": ('CLIP',)
      }
    }
  #
  @classmethod
  def IS_CHANGED(s, clip):
    return True
  #
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("CLIP", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, clip):
    return(clip,)

class supaidauen_passthrough_MASK:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "mask": ('MASK',)
      }
    }
  #
  @classmethod
  def IS_CHANGED(s, mask):
    return True
  #
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("MASK", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, mask):
    return(mask,)

class supaidauen_passthrough_STRING:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "string": ('STRING',)
      }
    }
  #
  @classmethod
  def IS_CHANGED(s, string):
    return True
  #
  #
  @classmethod
  def VALIDATE_INPUTS(s, **kwargs):
    return True
  #
  RETURN_TYPES = ("STRING", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, string):
    return(string,)
