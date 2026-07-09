import re

class supaidauen_ZippedPromptFromTextAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": (
                        "global_negative:\nblurry\nlow quality\n---\n"
                        "name: example\npositive:\na cat\n---\n"
                        "name: example2\npositive:\na dog\nnegative:\nfog"
                    )
                }),
            }
        }

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse"
    CATEGORY = "utils/prompt"

    # --- helpers ---

    def split_blocks(self, text):
        return re.split(r"\n\s*-{3,}\s*\n", text.replace("\r\n", "\n"))

    def extract_block(self, text, key):
        """
        Extract multiline block:
        key:
        line1
        line2
        until next key or end
        """
        pattern = rf"{key}:\s*\n(.*?)(?=\n\w+:|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_inline(self, text, key):
        pattern = rf"{key}:\s*(.*)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    # --- main parser ---

    def parse(self, text):

        blocks = self.split_blocks(text)

        global_negative = ""
        prompts = []

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            # detect global block
            if block.lower().startswith("global_negative"):
                gn = self.extract_block(block, "global_negative")
                if gn:
                    global_negative = gn
                continue

            # extract fields
            name = self.extract_inline(block, "name") or f"prompt_{len(prompts)}"

            positive = self.extract_block(block, "positive")
            if not positive:
                positive = self.extract_inline(block, "positive") or ""

            negative = self.extract_block(block, "negative")
            if not negative:
                negative = self.extract_inline(block, "negative")

            # fallback to global negative
            if not negative:
                negative = global_negative

            prompts.append((positive, negative, name))

        return (prompts,)
