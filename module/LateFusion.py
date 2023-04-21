import torch
import torch.nn as nn
import time

from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import soundfile as sf
import torch
from jiwer import wer


from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

import torch
import torch.nn as nn
from transformers import BertModel

import time


class KoBERTEmotionRecognition(nn.Module):
    def __init__(self):
        super(KoBERTEmotionRecognition, self).__init__()

        self.encoder = BertModel.from_pretrained("monologg/kobert")
        self.mlp = nn.Linear(in_features=768, out_features=7)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, ids, mask):
        encode_out = self.encoder(ids, mask)
        out = self.dropout(encode_out.pooler_output)
        out = self.mlp(out)

        return out, encode_out.pooler_output

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'KoBert_Classifier'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        # self.linear = nn.Linear(config.hidden_size, 768)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x



class Wav2VecEmotionRecognition(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # self.pooling_mode = config.pooling_mode
        self.config = config
        self.config.mask_time_length = 1        # test for error fix
        self.pooling_mode = "mean"
        self.num_labels = 7
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs
    
    def forward(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        return logits, hidden_states

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'Wav2Vec2_Classifier'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))


class LatefusionModule(nn.Module):
    def __init__(self, model_name_or_path, config):
        super(LatefusionModule, self).__init__()

        self.text_model = KoBERTEmotionRecognition()
        self.wav_model = Wav2VecEmotionRecognition.from_pretrained(model_name_or_path, config=config)

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Sequential(
        nn.Linear(1024 + 768, 768),
        nn.Tanh(),
        nn.Linear(768,7)
        )


    def forward(self, 
            t_ids, 
            t_masks, 
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None):
        
        _, t_hidden = self.text_model(t_ids, t_masks)
        _, w_hidden = self.wav_model(input_values, attention_mask)

        cat_res = torch.cat((t_hidden, w_hidden), dim=1)

        out = self.dropout(cat_res)
        out = self.linear(out)

        return out, t_hidden, w_hidden, cat_res

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'Wav2Vec2_Classifier'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))