import csv
import sys
import os
import collections
import torch
from torch.utils.data import TensorDataset
from transformers import BasicTokenizer
import logging
logger = logging.getLogger(__name__)

class MultiChoiceExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 ending_4,
                 ending_5,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
            ending_4,
            ending_5,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "swag_id: {}".format(self.swag_id),
            "context_sentence: {}".format(self.context_sentence),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
            "ending_4: {}".format(self.endings[4]),
            "ending_5: {}".format(self.endings[5]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)

    @classmethod
    def truncate_seq_pair(cls, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class MultiChoiceInputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

    @classmethod
    def select_field(cls, features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, pico=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.pico = pico
        self.labels = labels

    @classmethod
    def truncate_seq_pair(cls, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    # def __init__(self, input_ids, input_mask, segment_ids, pico_ids, label_ids):
    def __init__(self, input_ids, input_mask, segment_ids, pico_ids, pico_segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pico_ids = pico_ids
        self.pico_segment_ids = pico_segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                    logger=None,
                                    forSequenceTagging=False,
                                    min_seq_length=None,
                                    cls_token='[CLS]',
                                    sep_token_extra=False,
                                    sep_token = '[SEP]'):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_b = None
            if forSequenceTagging:
                tokens_a, pico_a, labels = tokenizer.tokenize_with_label_extension(example.text_a, example.pico, example.labels, copy_previous_label=True)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
                    pico_a = pico_a[:(max_seq_length-2)]
                    labels = labels[:(max_seq_length - 2)]
                labels = ["X"] + labels + ["X"]
                pico_a =  ["X"] + pico_a + ["X"]
            else:
                pico_a = example.pico[0]
                pico_b = example.pico[1]

                tokens_a, pico_a = tokenizer.tokenize_with_pico(example.text_a, pico_a)
                # tokens_a = tokenizer.tokenize(example.text_a)

                if example.text_b:
                    tokens_b, pico_b = tokenizer.tokenize_with_pico(example.text_b, pico_b)
                    # tokens_b = tokenizer.tokenize(example.text_b)

                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                    InputExample.truncate_seq_pair(pico_a, pico_b, max_seq_length - 3)
                else:
                    # Account for [CLS] and [SEP] with "- 2"
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
                        pico_a = pico_a[:(max_seq_length -2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = [cls_token] + tokens_a + [sep_token]
            #wrong - need different tags here.
            picos = ['X'] + pico_a + ['X']

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                picos += 'X'

            segment_ids = [0] * len(tokens)
            pico_segment_ids = [0] * len(picos)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                picos += pico_b + ['X']
                segment_ids += [1] * (len(tokens_b) + 1)
                pico_segment_ids += [1] * (len(pico_b) + 1)


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pico_ids = self.convert_pico_to_ids(picos)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            pad_size = (max_seq_length - len(input_ids))
            padding = [0] * pad_size
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            picos += ['X'] * pad_size
            # pico_ids += self.convert_pico_to_ids(['X']) * (max_seq_length - len(input_ids))
            pico_segment_ids += padding

            if forSequenceTagging:
                label_ids = self.convert_labels_to_ids(labels)
                label_ids += padding
                pico_ids = self.convert_pico_to_ids(pico_a)
                pico_ids += padding
                assert len(label_ids) == max_seq_length
                assert len(pico_ids) == max_seq_length
            else:
                label_list = self.get_labels()
                label_map = {label: i for i, label in enumerate(label_list)}
                label_ids = label_map[example.labels]
                pico_ids = self.convert_pico_to_ids(picos)


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if forSequenceTagging:
                    logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                    logger.info("pico_ids: %s" % " ".join([str(x) for x in pico_ids]))
                else:
                    logger.info("pico_segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    logger.info("pico_ids: %s" % " ".join([str(x) for x in pico_ids]))
                    logger.info("label_id: %s" % " ".join(str(label_ids)))

            if forSequenceTagging:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  pico_ids=pico_ids,
                                  pico_segment_ids=pico_segment_ids,
                                  label_ids=label_ids))
                #changes to InputFeatures line 141
            else:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  pico_ids = pico_ids,
                                  pico_segment_ids=pico_segment_ids,
                                  label_ids=label_ids))
        return features

    @classmethod
    def features_to_dataset(cls, feature_list, isMultiChoice=None):

        if isMultiChoice:
            all_input_ids = torch.tensor(MultiChoiceInputFeatures.select_field(feature_list, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(MultiChoiceInputFeatures.select_field(feature_list, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(MultiChoiceInputFeatures.select_field(feature_list, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in feature_list], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        else:
            all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
            all_pico_ids = torch.tensor([f.pico_ids for f in feature_list], dtype=torch.long)
            all_pico_segment_ids = torch.tensor([f.pico_segment_ids for f in feature_list], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pico_ids, all_pico_segment_ids, all_label_ids)
            # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_conll(cls, input_file, token_column=1, label_column=4, replace=None):
        """Reads a conll type file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines.append("\n") #workaround adding a stop criteria for last sentence iteration

            sentences = []
            try:
                lines[0].split('\t')[label_column]
            except IndexError as err:
                print('Label column', err)
                raise

            tokenizer = BasicTokenizer()
            sent_tokens = []
            sent_pico = []
            sent_labels = []

            for line in lines:

                line = line.split('\t')

                if len(line) < 2:
                    assert len(sent_tokens) == len(sent_labels)
                    if sent_tokens == []:
                        continue

                    if replace == None:
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    else:
                        sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                        sentences.append([' '.join(sent_tokens), sent_pico, sent_labels])
                    sent_tokens = []
                    sent_pico = []
                    sent_labels = []
                    continue

                token = line[token_column]
                pico = line[label_column-1]
                label = line[label_column].replace('\n', '')
                tokenized = tokenizer.tokenize(token)

                if len(tokenized) > 1:

                    for i in range(len(tokenized)):
                        if 'B-' in label:
                            if i < 1:
                                sent_tokens.append(tokenized[i])
                                sent_pico.append(pico)
                                sent_labels.append(label)
                            else:
                                sent_tokens.append(tokenized[i])
                                #sent_labels.append(label.replace('B-', 'I-')) #if only the first token should be B-
                                sent_pico.append(pico)
                                sent_labels.append(label)
                        else:
                            sent_tokens.append(tokenized[i])
                            sent_pico.append(pico)
                            sent_labels.append(label)

                else:
                    sent_tokens.append(tokenized[0])
                    sent_pico.append(pico)
                    sent_labels.append(label)

        return sentences

class PicoDataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                    logger=None,
                                    forSequenceTagging=False,
                                    min_seq_length=None,
                                    cls_token='[CLS]',
                                    sep_token_extra=False,
                                    sep_token = '[SEP]'):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_b = None
            if forSequenceTagging:
                tokens_a, labels = tokenizer.tokenize_with_label_extension(example.text_a, example.labels, copy_previous_label=True)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
                    labels = labels[:(max_seq_length - 2)]
                labels = ["X"] + labels + ["X"]
                # pico_a =  ["X"] + pico_a + ["X"]
            else:
                # pico_a = example.pico[0]
                # pico_b = example.pico[1]

                # tokens_a, pico_a = tokenizer.tokenize_with_pico(example.text_a, pico_a)
                tokens_a = tokenizer.tokenize(example.text_a)

                if example.text_b:
                    # tokens_b, pico_b = tokenizer.tokenize_with_pico(example.text_b, pico_b)
                    tokens_b = tokenizer.tokenize(example.text_b)

                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                    # InputExample.truncate_seq_pair(pico_a, pico_b, max_seq_length - 3)
                else:
                    # Account for [CLS] and [SEP] with "- 2"
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
                        # pico_a = pico_a[:(max_seq_length -2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = [cls_token] + tokens_a + [sep_token]
            #wrong - need different tags here.
            # picos = ['X'] + pico_a + ['X']

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                # picos += 'X'

            segment_ids = [0] * len(tokens)
            # pico_segment_ids = [0] * len(picos)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                # picos += pico_b + ['X']
                segment_ids += [1] * (len(tokens_b) + 1)
                # pico_segment_ids += [1] * (len(pico_b) + 1)


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # pico_ids = self.convert_pico_to_ids(picos)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            pad_size = (max_seq_length - len(input_ids))
            padding = [0] * pad_size
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            # picos += ['X'] * pad_size
            # pico_ids += self.convert_pico_to_ids(['X']) * (max_seq_length - len(input_ids))
            # pico_segment_ids += padding

            if forSequenceTagging:
                label_ids = self.convert_labels_to_ids(labels)
                label_ids += padding
                # pico_ids = self.convert_pico_to_ids(pico_a)
                # pico_ids += padding
                assert len(label_ids) == max_seq_length
                # assert len(pico_ids) == max_seq_length
            else:
                label_list = self.get_labels()
                label_map = {label: i for i, label in enumerate(label_list)}
                label_ids = label_map[example.labels]
                # pico_ids = self.convert_pico_to_ids(picos)


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if forSequenceTagging:
                    logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                    # logger.info("pico_ids: %s" % " ".join([str(x) for x in pico_ids]))
                else:
                    # logger.info("pico_segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    # logger.info("pico_ids: %s" % " ".join([str(x) for x in pico_ids]))
                    logger.info("label_id: %s" % " ".join(str(label_ids)))

            if forSequenceTagging:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  pico_ids=None,
                                  pico_segment_ids=None,
                                  label_ids=label_ids))
                #changes to InputFeatures line 141
            else:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  pico_ids = None,
                                  pico_segment_ids=None,
                                  label_ids=label_ids))
        return features

    @classmethod
    def features_to_dataset(cls, feature_list, isMultiChoice=None):
        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_conll(cls, input_file, token_column=0, label_column=4, replace=None):
        """Reads a conll type file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            # lines.append("\n") #workaround adding a stop criteria for last sentence iteration

            sentences = []
            try:
                lines[0].split(' ')[label_column]
            except IndexError as err:
                print('Label column', err)
                raise

            tokenizer = BasicTokenizer()
            sent_tokens = []
            sent_labels = []

            for line in lines:

                line = line.split(' ')

                if len(line) < 2:
                    assert len(sent_tokens) == len(sent_labels)
                    if sent_tokens == []:
                        continue

                    if replace == None:
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    else:
                        sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    sent_tokens = []
                    sent_labels = []
                    continue

                token = line[token_column]
                label = line[label_column].replace('\n', '')
                tokenized = tokenizer.tokenize(token)

                if len(tokenized) > 1:

                    for i in range(len(tokenized)):
                        if 'B-' in label:
                            if i < 1:
                                sent_tokens.append(tokenized[i])
                                sent_labels.append(label)
                            else:
                                sent_tokens.append(tokenized[i])
                                #sent_labels.append(label.replace('B-', 'I-')) #if only the first token should be B-
                                sent_labels.append(label)
                        else:
                            sent_tokens.append(tokenized[i])
                            sent_labels.append(label)

                else:
                    sent_tokens.append(tokenized[0])
                    sent_labels.append(label)

        return sentences

#todo: check _read_conll - possibly do different DataProcessor class as PicoDataProcessor
class ArgMinPicoSecTagProcessor(PicoDataProcessor):
    def __init__(self):
        self.labels = ["X", "N", "1_i", "1_o", "1_p"]
        self.label_map = self._create_label_map(self.labels)

    def _create_label_map(self, labels):
        label_map = collections.OrderedDict()
        for i, label in enumerate(labels):
            label_map[label] = i
        return label_map

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "p1_all_train.txt"), label_column=2), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "p1_all_dev.txt"), label_column=2), "dev")

    def get_test_examples(self, data_dir, setname="p1_all_gold.txt"):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, setname), label_column=2), "test")

    def convert_labels_to_ids(self, labels):
        idx_list = []
        for label in labels:
            idx_list.append(self.label_map[label])
        return idx_list

    def convert_ids_to_labels(self, idx_list):
        labels_list = []
        for idx in idx_list:
            labels_list.append([key for key in self.label_map.keys() if self.label_map[key] == idx][0])
        return labels_list

    def get_labels(self):
        """ See base class."""
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            labels = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, pico = None, labels=labels))
        return examples
    
class ArgMinSeqTagProcessor(DataProcessor):
    """
    Processor for argumentation mining sequence tagging task
    Created: 2025-02-16 19:35:31 UTC
    Author: kavish2003
    """
    
    def __init__(self):
        # Define labels in proper order for CRF
        self.labels = ["O", "I-Claim", "B-Claim", "X"]  # 4 labels total
        self.pico_labels = ["X", "N", "1_i", "1_o", "1_p"]
        
        # Create label mappings
        self.label_map = {
            "O": 0,
            "I-Claim": 1,
            "B-Claim": 2,
            "X": 3,
            # Map other labels to these basic categories
            "I-MajorClaim": 1,  # Map to I-Claim
            "B-MajorClaim": 2,  # Map to B-Claim
            "I-Premise": 1,     # Map to I-Claim
            "B-Premise": 2,     # Map to B-Claim
            "[CLS]": 3,
            "[SEP]": 3,
            "PAD": 0
        }
        
        self.pico_label_map = {
            "X": 0,
            "N": 1,
            "1_i": 2,
            "1_o": 3,
            "1_p": 4,
            "[CLS]": 0,
            "[SEP]": 0,
            "PAD": 0
        }

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "train_agg.conll")), 
            "train"
        )

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "dev_agg.conll")),
            "dev"
        )

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "test_agg.conll")),
            "test"
        )

    def get_labels(self):
        """Gets the list of labels."""
        return self.labels

    def get_pico_labels(self):
        """Gets the list of PICO labels."""
        return self.pico_labels

    def _read_conll(self, filepath):
        """
        Reads a CONLL formatted file
        Args:
            filepath: Path to the file
        Returns:
            List of [tokens, pico_labels, labels] lists
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CoNLL file not found at: {filepath}")

        logger.info(f"Reading CONLL file: {filepath}")
        
        sentences = []
        current_tokens = []
        current_pico = []
        current_labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith("-DOCSTART-"):
                    if current_tokens:
                        sentences.append([
                            current_tokens,
                            current_pico,
                            current_labels
                        ])
                        current_tokens = []
                        current_pico = []
                        current_labels = []
                    continue

                # Split by tab or multiple spaces
                parts = line.split('\t') if '\t' in line else line.split()
                
                if len(parts) >= 4:  # Format: index token pico label
                    token = parts[1]  # Token is in second column
                    pico = parts[2]   # PICO label is in third column
                    label = parts[3]  # Sequence label is in fourth column
                    
                    current_tokens.append(token)
                    current_pico.append(pico)
                    current_labels.append(label)

        # Add the last sentence if it exists
        if current_tokens:
            sentences.append([
                current_tokens,
                current_pico,
                current_labels
            ])

        logger.info(f"Read {len(sentences)} sentences from {filepath}")
        return sentences

    def _create_examples(self, sentences, set_type):
        """
        Creates examples for the training, dev and test sets.
        Args:
            sentences: List of [tokens, pico_labels, labels] lists
            set_type: train, dev, or test
        """
        examples = []
        for (i, sentence) in enumerate(sentences):
            guid = f"{set_type}-{i}"
            tokens = sentence[0]
            pico = sentence[1]
            labels = sentence[2]

            # Verify lengths match
            if not (len(tokens) == len(pico) == len(labels)):
                logger.warning(f"Skipping malformed sentence {guid}: length mismatch")
                continue

            examples.append(
                InputExample(
                    guid=guid,
                    text_a=tokens,
                    text_b=None,
                    pico=pico,
                    labels=labels
                )
            )

            if i < 2:  # Debug first two examples
                logger.info(f"\nExample {guid}:")
                logger.info(f"Tokens: {tokens[:5]}...")
                logger.info(f"PICO: {pico[:5]}...")
                logger.info(f"Labels: {labels[:5]}...")

        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                   logger=None, forSequenceTagging=True,
                                   min_seq_length=None):
        """Converts examples to features suitable for model input."""
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens = example.text_a
            pico_labels = example.pico
            labels = example.labels
            
            # Tokenize and align labels
            all_subtokens = []
            all_pico_labels = []
            all_labels = []
            
            for idx, (token, pico, label) in enumerate(zip(tokens, pico_labels, labels)):
                subtokens = tokenizer.tokenize(token)
                if not subtokens:
                    subtokens = [token]
                
                all_subtokens.extend(subtokens)
                
                # Handle PICO labels for subtokens
                all_pico_labels.extend([pico] * len(subtokens))
                
                # Handle sequence labels for subtokens
                if len(subtokens) == 1:
                    # Single token
                    if idx == 0 or labels[idx-1] == "O":  # Start of sequence or after O
                        if label.startswith("I-"):
                            all_labels.append("B-" + label[2:])
                        else:
                            all_labels.append(label)
                    else:
                        all_labels.append(label)
                else:
                    # Multiple subtokens
                    if label.startswith("B-") or (label.startswith("I-") and (idx == 0 or labels[idx-1] == "O")):
                        all_labels.append("B-" + label[2:])
                        all_labels.extend(["I-" + label[2:]] * (len(subtokens) - 1))
                    else:
                        all_labels.extend([label] * len(subtokens))
            
            # Truncate if needed
            if len(all_subtokens) > max_seq_length - 2:  # Account for [CLS] and [SEP]
                all_subtokens = all_subtokens[:(max_seq_length - 2)]
                all_pico_labels = all_pico_labels[:(max_seq_length - 2)]
                all_labels = all_labels[:(max_seq_length - 2)]
            
            # Add [CLS] and [SEP]
            tokens = ["[CLS]"] + all_subtokens + ["[SEP]"]
            pico_labels = ["X"] + all_pico_labels + ["X"]
            labels = ["[CLS]"] + all_labels + ["[SEP]"]
            
            # Convert to ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pico_ids = [self.pico_label_map.get(p, self.pico_label_map["X"]) for p in pico_labels]
            label_ids = [self.label_map.get(l, self.label_map["O"]) for l in labels]
            
            # Create attention mask and segment ids
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            
            # Zero-pad up to the sequence length
            padding_length = max_seq_length - len(input_ids)
            
            input_ids += [0] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [0] * padding_length
            label_ids += [self.label_map["PAD"]] * padding_length
            pico_ids += [self.pico_label_map["PAD"]] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(pico_ids) == max_seq_length

            if ex_index < 5 and logger:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("pico_ids: %s" % " ".join([str(x) for x in pico_ids]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    pico_ids=pico_ids,
                    pico_segment_ids=segment_ids,
                    label_ids=label_ids
                )
            )

        return features

class ArgMinRelClassProcessor(DataProcessor):
    """Processor for the RCT data set (for training)."""
    def __init__(self):
        self.pico_labels = ["X", "N", "1_i", "1_o", "1_p"]
        self.pico_label_map = self._create_label_map(self.pico_labels)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train_relations.tsv")), "train")
            self._read_tsv(os.path.join(data_dir, "train_relations_pico.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "dev_relations.tsv")), "dev")
            self._read_tsv(os.path.join(data_dir, "dev_relations_pico.tsv")), "dev")

    # def get_test_examples(self, data_dir, setname="test_relations.tsv"):
    def get_test_examples(self, data_dir, setname="test_relations_pico.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, setname)), "test")

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__Support", "__label__Attack"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip first line (e.g. PE dataset)
            #if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            pico = [line[3],line[4]]
            #dealing with partial labels / tags (removing them to full tags)
            if 'Partial' in label:
                label = ''.join(label.split('Partial-'))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, pico=pico, labels=label))
        return examples

    def _create_label_map(self, labels):
        label_map = collections.OrderedDict()
        for i, label in enumerate(labels):
            label_map[label] = i
        return label_map

    def convert_pico_to_ids(self, pico):
        idx_list = []
        for l in pico:
            idx_list.append(self.pico_label_map[l])
        return idx_list

class ArgMinRelClassForMultiChoiceProcessor(ArgMinRelClassProcessor):
    """Processor for the RCT data set (for the relation classification in the multiple choice training)."""

    def get_labels(self):
        """See base class."""
        return ["__label__Support", "__label__Attack"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):


            if line[0] == "__label__noRel":
                continue

            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
        return examples


class ArgMinMultiChoiceLinkProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labelmap = {
            "NoRelation": 2,
            "Support": 0,
            "Attack": 1,
            "Partial-Attack": 1
        }


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_mc.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mc.tsv")), "dev")

    def get_test_examples(self, data_dir, setname="test_mc.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, setname)), "test")

    def get_labels(self):
        """ See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        examples = []

        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            context_sentence = line[0]
            ending_0 = line[1]
            ending_1 = line[2]
            ending_2 = line[3]
            ending_3 = line[4]
            ending_4 = line[5]
            ending_5 = line[6]
            #label = int(line[7])
            label = (int(line[7]), self.labelmap[line[8]])
            examples.append(MultiChoiceExample(
                swag_id=guid,
                context_sentence=context_sentence,
                ending_0=ending_0,
                ending_1=ending_1,
                ending_2=ending_2,
                ending_3=ending_3,
                ending_4=ending_4,
                ending_5=ending_5,
                label=label
            ))
        return examples

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length, logger=None):
        """Loads a data file into a list of `InputBatch`s."""

        # Swag is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        for example_index, example in enumerate(examples):
            context_tokens = tokenizer.tokenize(example.context_sentence)

            choices_features = []
            for ending_index, ending in enumerate(example.endings):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                ending_tokens = tokenizer.tokenize(ending)
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                MultiChoiceExample.truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if example_index < 3 and logger is not None:
                logger.info("*** Example ***")
                logger.info("example_id: {}".format(example.swag_id))
                for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                    logger.info("choice_idx: {}".format(choice_idx))
                    logger.info("tokens: {}".format(' '.join(tokens)))
                    logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                    logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                    logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                    logger.info("label: {}".format(label))

            features.append(
                MultiChoiceInputFeatures(
                    example_id=example.swag_id,
                    choices_features=choices_features,
                    label=label
                )
            )

        return features


processors = {
    "seqtag": ArgMinSeqTagProcessor,
    "relclass": ArgMinRelClassProcessor,
    "multichoice": (ArgMinRelClassForMultiChoiceProcessor, ArgMinMultiChoiceLinkProcessor),
    "picoseqtag": ArgMinPicoSecTagProcessor,
}

output_modes = {
    "seqtag": "sequencetagging",
    "relclass": "classification",
    "multichoice": "classification",
    "picoseqtag": "sequencetagging"
}