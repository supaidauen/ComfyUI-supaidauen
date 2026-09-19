import random
import re

class RandomPromptText:
    """
    Text box that resolves {option1|option2|option3} groups into a single
    randomly-chosen option, seeded for repeatability.

    Example:
        "This is a drawing of a {cat|dog|bird} that is {blue|red|brown}."
        -> "This is a drawing of a dog that is brown."

    Supports nested groups, e.g. "{a|{b|c}}", by resolving innermost
    groups first and repeating until no braces remain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "This is a drawing of a {cat|dog|bird} that is {blue|red|brown}."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,  # adds randomize/increment/fixed control in the UI
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "seed")
    FUNCTION = "process"
    CATEGORY = "text"

    # matches the innermost {...} group (no nested braces inside it)
    _PATTERN = re.compile(r"\{([^{}]*)\}")

    def process(self, text, seed):
        rng = random.Random(seed)
        result = text

        while True:
            match = self._PATTERN.search(result)
            if not match:
                break
            options = match.group(1).split("|")
            options = [o for o in options] or [""]
            choice = rng.choice(options)
            result = result[:match.start()] + choice + result[match.end():]

        return (result, seed)


NODE_CLASS_MAPPINGS = {
    "RandomPromptText": RandomPromptText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPromptText": "Random Prompt Text (Seeded)",
}
