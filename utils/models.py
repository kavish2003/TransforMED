import copy
import torch
import numpy as np
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
from utils.layers import MultiHeadedAttention, PositionwiseFeedForward, Encoder, EncoderLayer, Decoder, DecoderLayer

class BertForSequenceTagging(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        pico_embeddings_size = 100  # Randomly initialized embedding size
        self.bert = BertModel(config)

        self.pico_embeddings = nn.Embedding(pico_embeddings_size, pico_embeddings_size)  # Pico embedding layer

        self.crf = CRF(4, batch_first=True)  # CRF for sequence tagging
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        N = 4  # Number of layers
        h = 4  # Number of heads
        dropout_value = 0.1
        d_model = config.hidden_size + pico_embeddings_size
        d_ff = 2048
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value), N)

        self.classifier_bienc = nn.Linear(2 * d_model, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pico=None,  # Pico tensor, can be None
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

        sequence_output = outputs[0]  # Shape: (batch_size, sequence_length, hidden_size)

        # Ensure pico is not None
        if pico is None:
            batch_size = sequence_output.shape[0]
            pico = torch.zeros(batch_size, dtype=torch.long, device=sequence_output.device)

        pico_input = self.pico_embeddings(pico)  # Shape: (batch_size, pico_embedding_size)
        pico_input = pico_input.unsqueeze(1).expand(-1, sequence_output.shape[1], -1)  # Expand to match sequence_output

        latent_input = torch.cat([sequence_output, pico_input], dim=-1)  # Now both have shape (batch, seq_len, d_model)
        forward_encode = self.encoder(latent_input, None)
        backward_encode = self.encoder(torch.flip(latent_input, [-1]), None)
        encode = torch.cat([forward_encode, backward_encode], dim=-1)
        emissions = self.classifier_bienc(encode)

        if labels is not None:
            loss = self.crf(emissions, labels)
            path = self.crf.decode(emissions)
            path = torch.LongTensor(path).to(emissions.device)
            return (-1 * loss, emissions, path)
        else:
            path = self.crf.decode(emissions)
            path = torch.LongTensor(path).to(emissions.device)
            return path
        
        
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
