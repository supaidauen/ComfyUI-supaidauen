import re

class supaidauen_StructuredBatchConditioningGlobalNegative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "prompt: ...\n---\nprompt: ..."
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("images", "positive_batch", "negative_batch")
    FUNCTION = "process"
    CATEGORY = "utils/structured"

    # --- parsing ---
    def split_blocks(self, text):
        text = text.replace("\r\n", "\n")
        return re.split(r"\n\s*---{3,}\s*\n", text)

    def parse_block(self, block):
        data = {}
        for line in block.strip().split("\n"):
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip().lower()] = v.strip()
        return data

    def parse_all(self, text):
        return [
            self.parse_block(b)
            for b in self.split_blocks(text)
            if b.strip()
        ]

    # --- CLIP encode ---
    def encode(self, clip, text):
        if not text:
            text = ""

        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    # --- main ---
    def process(self, images, text, negative, clip):
        if not isinstance(images, list):
            images = [images]

        entries = self.parse_all(text)

        if len(images) != len(entries):
            raise ValueError(
                f"Mismatch: {len(images)} images vs {len(entries)} prompt blocks"
            )

        # Encode GLOBAL negative ONCE
        neg_cond = self.encode(clip, negative)

        positive_batch = []
        negative_batch = []

        for entry in entries:
            prompt = entry.get("prompt", "")
            pos_cond = self.encode(clip, prompt)

            positive_batch.append(pos_cond)
            negative_batch.append(neg_cond)

        return (images, positive_batch, negative_batch)