from transformers import BertTokenizer
import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
import torch

logger = logging.getLogger(__name__)

class ExtendedBertTokenizer(BertTokenizer):
    """BertTokenizer which extends a token-based list of labels to WordPiece sub-token-based list."""
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        logger.info("Initializing ExtendedBertTokenizer")

    def tokenize_with_label_extension(self, 
                                    text: str, 
                                    pico: List[str], 
                                    labels: List[str], 
                                    copy_previous_label: bool = False, 
                                    extension_label: str = 'X') -> Tuple[List[str], List[str], List[str]]:
        """
        Tokenize text and extends the label list to match the length of the tokenizer output.
        
        Args:
            text: Input text to tokenize
            pico: List of PICO labels
            labels: List of class labels
            copy_previous_label: If True, copies previous label for subtokens
            extension_label: Label to use for subtokens if copy_previous_label is False
            
        Returns:
            Tuple of (tokenized_text, extended_pico_labels, extended_class_labels)
        """
        tok_text = self.tokenize(text)
        
        logger.debug(f"Original text: {text}")
        logger.debug(f"Tokenized text: {tok_text}")
        logger.debug(f"Original labels: {labels}")
        logger.debug(f"Original PICO: {pico}")

        for i in range(0, len(tok_text)):
            if '##' in tok_text[i]:
                if copy_previous_label:
                    labels.insert(i, labels[i-1])
                    pico.insert(i, pico[i-1])
                else:
                    labels.insert(i, extension_label)
                    pico.insert(i, extension_label)

        logger.debug(f"Extended labels: {labels}")
        logger.debug(f"Extended PICO: {pico}")
        return tok_text, pico, labels

    def tokenize_with_pico(self, text: str, pico: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize text while maintaining PICO labels alignment.
        
        Args:
            text: Input text to tokenize
            pico: Space-separated PICO labels
            
        Returns:
            Tuple of (tokenized_text, aligned_pico_labels)
        """
        text = text.split(' ')
        pico = pico.split(' ')
        new_text = []
        new_pico = []
        
        for i, token in enumerate(text):
            token = self.tokenize(token)
            for t in token:
                new_text.append(t)
                new_pico.append(pico[i])

        logger.debug(f"Tokenized text: {new_text}")
        logger.debug(f"Aligned PICO: {new_pico}")
        return new_text, new_pico

    def batch_to_conll(self, 
                      input_ids: torch.Tensor, 
                      label_ids: np.ndarray, 
                      gold_label: Optional[np.ndarray] = None, 
                      processor: Optional[object] = None, 
                      output_file: Optional[str] = None) -> None:
        """
        Convert batch predictions to CONLL format
        
        Args:
            input_ids: Tensor of input token IDs
            label_ids: Array of predicted label IDs
            gold_label: Optional array of true label IDs
            processor: Processor instance for label conversion
            output_file: Path to output file
            
        Raises:
            AssertionError: If input and label shapes don't match
            ValueError: If input validation fails
        """
        logger.info("\nWriting predictions to CONLL format")
        logger.debug(f"Input shape: {input_ids.shape}")
        logger.debug(f"Label shape: {label_ids.shape}")
        
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
            
        if not isinstance(label_ids, np.ndarray):
            raise ValueError("label_ids must be a numpy.ndarray")
            
        assert input_ids.shape == label_ids.shape, "Input and label shapes must match"

        sentences = []

        try:
            for i, _ in enumerate(input_ids):  # iterate over batch
                sentence = []
                y_pred = []

                for j, token_id in enumerate(input_ids[i].cpu().numpy()):
                    token = self._convert_id_to_token(token_id)

                    if processor is not None:
                        y_pred.append(processor.convert_ids_to_labels([int(label_ids[i][j])])[0])
                    else:
                        y_pred.append(label_ids[i][j])
                    logger.debug(f"Predicted label IDs: {label_ids[i]}")
                    logger.debug(f"Converted labels: {y_pred}")
                    sentence.append(token)

                if gold_label is not None:
                    if processor is not None:
                        y_true = processor.convert_ids_to_labels(list(gold_label[i]))
                    else:
                        y_true = gold_label[i]
                    sentences.append((sentence, y_pred, y_true))
                else:
                    sentences.append((sentence, y_pred))

            with open(output_file, "a", encoding='utf-8') as writer:
                for sentence, preds, labels in sentences:
                    for i, w in enumerate(sentence):
                        if w in {"[CLS]", "[SEP]", "[PAD]"}:
                            continue
                        writer.write(f"{w}\t{preds[i]}\t{labels[i]}\n")
                    writer.write("\n")
                    
            logger.info(f"Successfully wrote predictions to {output_file}")
            
        except Exception as e:
            logger.error(f"Error in batch_to_conll: {str(e)}")
            logger.error(f"Input IDs shape: {input_ids.shape}")
            logger.error(f"Label IDs shape: {label_ids.shape}")
            if gold_label is not None:
                logger.error(f"Gold label shape: {gold_label.shape}")
            raise


class ExtendedPicoBertTokenizer(ExtendedBertTokenizer):
    """BertTokenizer specialized for PICO sequence tagging"""
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        logger.info("Initializing ExtendedPicoBertTokenizer")

    def tokenize_with_label_extension(self, 
                                    text: str, 
                                    labels: List[str], 
                                    copy_previous_label: bool = False, 
                                    extension_label: str = 'X') -> Tuple[List[str], List[str]]:
        """
        Tokenize text and extends the label list to match the length of the tokenizer output.
        
        Args:
            text: Input text to tokenize
            labels: List of labels
            copy_previous_label: If True, copies previous label for subtokens
            extension_label: Label to use for subtokens if copy_previous_label is False
            
        Returns:
            Tuple of (tokenized_text, extended_labels)
        """
        tok_text = self.tokenize(text)
        
        logger.debug(f"Original text: {text}")
        logger.debug(f"Tokenized text: {tok_text}")
        logger.debug(f"Original labels: {labels}")

        for i in range(0, len(tok_text)):
            if '##' in tok_text[i]:
                if copy_previous_label:
                    labels.insert(i, labels[i-1])
                else:
                    labels.insert(i, extension_label)

        logger.debug(f"Extended labels: {labels}")
        return tok_text, labels
