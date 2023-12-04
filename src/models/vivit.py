from transformers import VivitForVideoClassification, VivitImageProcessor
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """
        Override forward method so that, when a list of segments for the same video is passed, the forward method is
        called for each segment and the results are averaged.
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :param labels:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param kwargs:
        :return:
        """
        # If input_ids is a list, then we are evaluating a list of segments for the same video
        if isinstance(input_ids, list):
            # Compute the forward pass for each segment and average the results
            outputs = []
            for segment_input_ids in input_ids:
                outputs.append(
                    self.vivit(
                        input_ids=segment_input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        **kwargs
                    )
                )
            # Average the results
            logits = torch.stack([output.logits for output in outputs]).mean(dim=0)
            loss = torch.stack([output.loss for output in outputs]).mean(dim=0)
            return self.vivit_image_processor.post_process(logits, loss)
        else:
            # If input_ids is not a list, then we are evaluating a single segment
            return self.vivit(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
