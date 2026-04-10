class EndAtStepsModelControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "end_at_step": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/control"

    def apply(self, model, end_at_step):
        model = model.clone()
        model.model_options = dict(model.model_options)
        model.model_options["end_at_step"] = end_at_step
        return (model,)