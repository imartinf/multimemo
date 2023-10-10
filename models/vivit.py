import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

class CustomVivit(nn.Module):
    """
    Custom Vivit model that allows finetuning
    """

    def __init__(self, model_name, num_classes, loss_fn=None, dropout_prob=0.1):
        """
        Custom Vivit model that allows finetuning

        :param model_name: The name of the model to load.
        :type model_name: str

        :param num_classes: The number of classes.
        :type num_classes: int

        :param loss_fn: The loss function to use.
        :type loss: torch.nn.modules.loss._Loss

        :param dropout_prob: The dropout probability to use.
        :type dropout_prob: float

        :return: None
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.dropout_prob = dropout_prob
        self.model = self.get_model()
        self.dropout = nn.Dropout(dropout_prob)
        # Classifier must have a dense, dropout and linear layer
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(self.model.config.hidden_size, self.num_classes)
        )
                                         

    # Make it work so that calling the model() calls the forward() method
    def __call__(self, input_ids, attention_mask, labels=None):
        return self.forward(input_ids, attention_mask, labels)


    def get_model(self):
        """
        Get the model

        :return: The model
        """
        return AutoModel.from_pretrained(self.model_name)
        
        
    
    def forward(self, pixel_values, attention_mask, labels=None):
        """
        Forward pass

        :param input_ids: The input ids.
        :type input_ids: torch.Tensor

        :param attention_mask: The attention mask.
        :type attention_mask: torch.Tensor

        :param labels: The labels.
        :type labels: torch.Tensor

        :return: The loss and logits
        """
        # Get the last hidden state
        last_hidden_state = self.model(pixel_values, attention_mask=attention_mask).last_hidden_state
        # Get the CLS token
        cls_token = last_hidden_state[:, 0, :]
        # Apply dropout
        cls_token = self.dropout(cls_token)
        # Apply the classifier
        logits = self.classifier(cls_token)
        # If labels are provided, compute the loss
        if labels is not None and self.loss_fn is not None:
            return self.loss_fn(logits, labels), logits
        else:
            return logits
