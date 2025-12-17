import re
from datetime import datetime as dt

class supaidauen_PromptConsolidator:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
      "optional": {
        "positve_g": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "positve_l": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "negative": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "option_replace":  ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "replace": ("STRING", {"multiline": True, "default": ""}),
        "split": ("STRING", {"multiline": False, "default": ";", "forceInput": False}),
      },
    }
  #
  RETURN_TYPES = ("STRING", "STRING", "STRING", )
  RETURN_NAMES = ("positive_g", "positive_l", "negative", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, positve_g="", positve_l="", negative="", option_replace='', replace='',split=';' ):
    r_w_o = option_replace.split(split)
    r_o = replace.split(split)
    combined_replace = list(itertools.zip_longest(r_w_o, r_o, fillvalue=''))
    for r_w,t_w in combined_replace:
      if r_w == "" or t_w == "":
        pass
      else:
        positve_g = positve_g.replace(t_w,r_w)
        positve_l = positve_l.replace(t_w,r_w)
        negative = negative.replace(t_w,r_w)
    return (positve_g,positve_l,negative)

class supaidauen_TextConcatenate:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
        "optional": {
        "text1": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "text2": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "separator": ("STRING", {"multiline": False, "default": ""}),
      },
    }
  #
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("STRING", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, text1="", text2="", separator=""):
      return (f"{text1}{separator}{text2}", )

class supaidauen_TextReplace:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
        "optional": {
        "text": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "replace_with": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
        "replace": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "split": ("STRING", {"multiline": False, "default": ";", "forceInput": False}),
      },
    }
  #
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("text", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, text="", replace_with="", replace="",split=';'):
      r_w_o = replace_with.split(split)
      r_o = replace.split(split)
      combined_replace = list(itertools.zip_longest(r_w_o, r_o, fillvalue=''))
      for r_w,t_w in combined_replace:
        if r_w == "" or t_w == "":
          pass
        else:
          text  = text.replace(t_w,r_w)
      text_return = text
      return (f"{text_return}", )

class supaidauen_TextWildcard:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "wildcard": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
      },
        "optional": {
          "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
        "hidden": {
          "unique_id": "UNIQUE_ID",
        },
    }
  #
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("text", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, wildcard="", seed=0, unique_id=0):
    if seed is not None:
      seed = seed + int(unique_id)
      random.seed(seed)

    pattern = re.compile(r"\{([^{}]+)\}")

    def pick_choice(match):
      choices = match.group(1).split("|")
      return random.choice(choices)

    text = pattern.sub(pick_choice, wildcard)
    return (f"{text}", )

class supaidauen_Text_w_Options_Replace_LoRA:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
        "optional": {
        "option_add":  ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "option_replace":  ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "wildcard": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "lora_15": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "lora_XL": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
        "replace": ("STRING", {"multiline": False, "default": ""}),
        "split": ("STRING", {"multiline": False, "default": ";", "forceInput": False}),
      },
    }
  #
  RETURN_TYPES = ("STRING", "STRING", "STRING" )
  RETURN_NAMES = ("wildcard_15", "wildcard_XL", "wildcard")
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"

  def doit(self, option_add='', option_replace='', wildcard='', lora_15='', lora_XL='', replace='',split=';'):
    BREAK = "\nBREAK\n"
    option_replace = option_replace.split(split)
    replace = replace.split(split)
    combined_replace = list(itertools.zip_longest(option_replace, replace, fillvalue=''))
    for replace_with,to_replace in combined_replace:
      if replace_with == "" or to_replace == "":
        pass
      else:
        wildcard = wildcard.replace(to_replace,replace_with)
    wildcard = wildcard+BREAK+option_add
    wildcard_15 = wildcard+BREAK+lora_15.replace(",","")
    wildcard_XL = wildcard+BREAK+lora_XL.replace(",","")
    return (wildcard_15, wildcard_XL, wildcard )

class supaidauen_TextConcatenateFileName:
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
      "optional": {
        "moniker": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "seed": ("INT", {"default": 0, "forceInput": True}),
        "sd15": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "sdXL": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "PNY": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "pose": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "separator": ("STRING", {"multiline": False, "default": "-"}),
      },
    }
  #
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("STRING", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  def doit(self, moniker="",PNY="",sd15="",sdXL="",pose="",separator="",seed=0):
    sd15 = sd15.replace(".safetensors","").replace(".pt","")
    if "\\" in sd15:
      sd15 = sd15.split("\\")[1]
    sdXL = sdXL.replace(".safetensors","").replace(".pt","")
    if "\\" in sdXL:
      sdXL = sdXL.split("\\")[1]
    PNY = PNY.replace(".safetensors","").replace(".pt","")
    if "\\" in PNY:
      PNY = PNY.split("\\")[1]
    padded_seed = f"{seed:016}"
    pose = pose.split("\\")[-1].split(".")[0]
    return (f"{moniker}{separator}{padded_seed}{separator}{PNY}{separator}{sd15}{separator}{sdXL}{separator}{pose}", )

class supaidauen_RunIDConcatenate :
  @ classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "string" : ("STRING", {"multiline": False, "default": "", "forceInput": True}),
        "separator" : ("STRING", {"multiline": False, "default": "", "forceInput": False}),
      },
      "optional": {},
    }
  #
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("string", )
  FUNCTION = "doit"
  CATEGORY = "supaidauen/Util"
  #
  #
  #
  @classmethod
  def IS_CHANGED(self, string, separator):
    return float('nan')
  #
  def doit(self, string='',separator=''):
    runid = dt.today().strftime('%Y%m%d%H%M%S')
    string = string+separator+runid
    return (string, )
