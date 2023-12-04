from transformers import VivitForVideoClassification, VivitImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput
import torch


class CustomVivit(VivitForVideoClassification):
    """
    Custom Vivit model that inherits from VivitForVideoClassification.
    The main goal of this class is to override forward so that, on eval, when a list of segments for
    the same video is passed, the forward method is called for each segment and the results are averaged.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vivit = VivitForVideoClassification(config)
        self.vivit_image_processor = VivitImageProcessor(config)
        self.vivit_image_processor.config = config

    def forward(self, video, **kwargs):
        """
        Override forward method so that, when a list of segments for the same video is passed, the forward method is
        called for each segment and the results are averaged.
        :param video: List of segments for the same video
        :param kwargs: Forward kwargs
        :return: A tuple with the logits and the loss
        """
        # If input_ids is a list, then we are evaluating a list of segments for the same video
        if isinstance(video, list):
            outputs = []
            for segment in video:
                # Check if the segment has 5 dimensions and if it has 4 add a new dimension
                if len(segment["pixel_values"].shape) == 4:
                    segment["pixel_values"] = torch.unsqueeze(segment["pixel_values"], 0)
                outputs.append(
                    self.vivit(
                        **segment,
                    )
                )
            # Average the results
            logits = torch.stack([output.logits for output in outputs]).mean(dim=0)
            if outputs[0].loss is None:
                loss = None
            else:
                loss = torch.stack([output.loss for output in outputs if output.loss is not None]).mean(dim=0)
            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )
        else:
            # If input_ids is not a list, then we are evaluating a single segment
            return self.vivit(
                **video,
            )
