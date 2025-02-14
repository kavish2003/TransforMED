import copy
import torch
import numpy as np
# from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
from utils.layers import MultiHeadedAttention, PositionwiseFeedForward, Encoder, EncoderLayer, Decoder, DecoderLayer
import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Validate label configuration
        if not hasattr(config, 'id2label') or not hasattr(config, 'label2id'):
            raise ValueError("Model config must include id2label and label2id mappings")
        if len(config.id2label) != config.num_labels:
            raise ValueError(f"Number of labels in id2label ({len(config.id2label)}) "
                           f"doesn't match num_labels ({config.num_labels})")
        
        self.num_labels = config.num_labels
        pico_embeddings_size = 100
        self.bert = BertModel(config)

        # Store label mappings
        self.label2id = config.label2id
        self.id2label = config.id2label

        # PICO embeddings
        self.pico_embeddings = nn.Embedding(pico_embeddings_size, pico_embeddings_size)

        # Update CRF to use num_labels from config
        self.crf = CRF(self.num_labels, batch_first=True)
        self.classifier = nn.Linear(2 * config.hidden_size, self.num_labels)

        # Transformer architecture parameters
        N = 4  # Number of layers
        h = 4  # Number of heads
        dropout_value = 0.1
        d_model = config.hidden_size + pico_embeddings_size
        d_ff = 2048
        c = copy.deepcopy
        
        # Initialize transformer components
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value), N)

        self.classifier_bienc = nn.Linear(2 * d_model, self.num_labels)
        
        self.init_weights()

    def _validate_labels(self, labels):
        """Validate that labels are within the expected range"""
        if labels is not None:
            valid_labels = set(range(self.num_labels))
            valid_labels.add(-100)  # Add padding index
            labels_set = set(labels.unique().cpu().numpy())
            invalid_labels = labels_set - valid_labels
            if invalid_labels:
                raise ValueError(f"Invalid label ids found: {invalid_labels}. "
                              f"Expected labels in range [0, {self.num_labels-1}] or -100")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pico=None,
        labels=None,
    ):
        # Input validation
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Validate labels if provided
        self._validate_labels(labels)

        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        batch_size, seq_length, hidden_size = sequence_output.shape

        # Handle PICO embeddings
        if pico is None:
            pico = torch.zeros(batch_size, dtype=torch.long, device=sequence_output.device)
        pico_input = self.pico_embeddings(pico)
        pico_input = pico_input.unsqueeze(1).expand(-1, seq_length, -1)

        # Combine BERT and PICO embeddings
        latent_input = torch.cat([sequence_output, pico_input], dim=-1)

        # Create attention mask for encoder if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=latent_input.device)

        # Bidirectional encoding
        forward_encode = self.encoder(latent_input, attention_mask)
        # Flip attention mask for backward pass
        backward_mask = torch.flip(attention_mask, [1]) if attention_mask is not None else None
        backward_encode = self.encoder(torch.flip(latent_input, [1]), backward_mask)
        backward_encode = torch.flip(backward_encode, [1])  # Flip back
        
        # Combine forward and backward encodings
        encode = torch.cat([forward_encode, backward_encode], dim=-1)
        
        # Get emissions for CRF
        emissions = self.classifier_bienc(encode)

        # Create mask for CRF
        crf_mask = attention_mask.bool() if attention_mask is not None else None
        if labels is not None:
            # Mask out padding tokens (-100)
            label_mask = (labels != -100)
            if crf_mask is not None:
                crf_mask = crf_mask & label_mask

        outputs = (emissions,)

        if labels is not None:
            # CRF loss calculation
            loss = -1 * self.crf(emissions, labels, mask=crf_mask)
            # Get best path
            path = self.crf.decode(emissions, mask=crf_mask)
            path = torch.LongTensor(path).to(emissions.device)
            
            outputs = (loss, emissions, path)
        else:
            # Inference mode
            path = self.crf.decode(emissions, mask=crf_mask)
            path = torch.LongTensor(path).to(emissions.device)
            outputs = path

        return outputs

    def decode(self, emissions, attention_mask=None):
        """Convenience method for getting predictions"""
        with torch.no_grad():
            mask = attention_mask.bool() if attention_mask is not None else None
            path = self.crf.decode(emissions, mask=mask)
            return torch.LongTensor(path).to(emissions.device)
        
        
class BertForSeqClass(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        pico_embeddings_size = 100

        self.bert = BertModel(config)

        self.pico_embeddings = nn.Embedding(5, pico_embeddings_size)

        N = 4  # Number of layers
        h = 4  # Number of heads
        dropout_value = 0.1
        d_model = config.hidden_size + pico_embeddings_size
        d_ff = 2 * d_model
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value), N)

        self.transformer_pooler = nn.Linear(d_model, d_model)
        self.transformer_pooler_activation = nn.Tanh()

        self.classifier = nn.Linear(d_model, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pico=None,
        labels=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_embeddings = outputs[0]

        if pico is None:
            batch_size = sequence_embeddings.shape[0]
            pico = torch.zeros(batch_size, dtype=torch.long, device=sequence_embeddings.device)

        pico_embeddings = self.pico_embeddings(pico)
        input_data = torch.cat([sequence_embeddings, pico_embeddings], dim=-1)

        latent_input = self.encoder(input_data, None)
        latent_input = self.decoder(latent_input, latent_input, None, None)

        latent_pooled = latent_input[:, 0]
        latent_pooled = self.transformer_pooler(latent_pooled)
        latent_pooled = self.transformer_pooler_activation(latent_pooled)

        logits = self.classifier(latent_pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class BertForPicoSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Debug prints
            print("\nDEBUG - Model Forward Pass:")
            print(f"Logits shape: {logits.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Loss: {loss.item()}")
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            print(f"Unique predictions: {torch.unique(preds, return_counts=True)}")
            
            outputs = (loss,) + outputs

        return outputs
