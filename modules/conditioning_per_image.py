class supaidauen_Conditioning_per_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompts": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING")
    RETURN_NAMES = ("image", "conditioning")
    FUNCTION = "get_conditioning"
    CATEGORY = "utils/pairing"

    def parse_prompts(self, prompts):
        # Already a list? Return as-is
        if isinstance(prompts, list):
            return prompts

        # Normalize line endings
        text = prompts.replace("\r\n", "\n")

        # Split on lines that are ONLY hashes (e.g. ###, ####, etc.)
        import re
        raw_blocks = re.split(r"\n#{3,}\n", text)

        # Clean each block
        parsed = []
        for block in raw_blocks:
            cleaned = block.strip()
            if cleaned:
                parsed.append(cleaned)

        return parsed

    def get_conditioning(self, images, prompts, clip, index):
        # Normalize inputs
        if not isinstance(images, list):
            images = [images]

        prompts = self.parse_prompts(prompts)

        if len(images) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(images)} images vs {len(prompts)} prompts"
            )

        if index >= len(images):
            raise IndexError(f"Index {index} out of range")

        image = images[index]
        prompt = prompts[index]

        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        conditioning = [[cond, {"pooled_output": pooled}]]

        return (image, conditioning)