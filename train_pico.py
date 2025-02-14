import argparse
import glob
import json
import logging
import os
import random
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter
from utils.data_processors import output_modes, processors
from utils.models import BertForSequenceTagging, BertForSeqClass, BertForPicoSequenceTagging
from utils.metrics import compute_metrics
from utils.tokenizer import ExtendedBertTokenizer, ExtendedPicoBertTokenizer
from label_config import LABEL_MAP, ID2LABEL
import torch
import logging
import os
import random
import numpy as np
from transformers import WEIGHTS_NAME

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert-seqtag": (
        lambda: BertConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=4,  # Number of labels in your dataset
            id2label={
                0: "I-Claim",
                1: "I-MajorClaim",
                2: "O",
                3: "I-Premise"
            },
            label2id={
                "I-Claim": 0,
                "I-MajorClaim": 1,
                "O": 2,
                "I-Premise": 3
            }
        ),
        BertForSequenceTagging,
        ExtendedBertTokenizer
    ),
    "bert-seqtag-pico": (
        lambda: BertConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=4,  # Number of labels in your dataset
            id2label={
                0: "I-Claim",
                1: "I-MajorClaim",
                2: "O",
                3: "I-Premise"
            },
            label2id={
                "I-Claim": 0,
                "I-MajorClaim": 1,
                "O": 2,
                "I-Premise": 3
            }
        ),
        BertForPicoSequenceTagging,
        ExtendedPicoBertTokenizer
    ),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model with improved label handling and monitoring """
    
    # Add label configuration validation
    print("\nValidating model configuration...")
    config = model.config
    label_map = {
        "Premise": 0,
        "Claim": 1,
        "MajorClaim": 2,
        "O": 3
    }
    id2label = {v: k for k, v in label_map.items()}
    
    # Validate model configuration
    if config.num_labels != len(label_map):
        raise ValueError(
            f"Model configured for {config.num_labels} labels but expected {len(label_map)}. "
            f"Please check model configuration."
        )
    
    # Initialize tensorboard writer
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Analyze dataset label distribution
    print("\nAnalyzing training data distribution...")
    all_labels = []
    for batch in DataLoader(train_dataset, batch_size=1):
        labels = batch[3].numpy()
        all_labels.extend(labels[labels != -100])  # Exclude padding
    
    label_counts = np.bincount(all_labels)
    print("Label distribution in training data:")
    for label_id, count in enumerate(label_counts):
        if label_id in id2label:
            print(f"{id2label[label_id]}: {count} ({count/len(all_labels)*100:.2f}%)")

    # Calculate class weights for balanced loss
    total_samples = sum(label_counts)
    class_weights = torch.FloatTensor([
        total_samples / (len(label_counts) * count) if count > 0 else 0
        for count in label_counts
    ]).to(args.device)
    
    # Initialize weighted loss function
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-100  # Ignore padding tokens
    )

    # Setup training parameters
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Calculate total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Load optimizer and scheduler states if exists
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and \
       os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Mixed precision training
    if args.fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Training logs
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(
        f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    # Training state
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_eval_f1 = 0.0
    
    # Resume from checkpoint if exists
    if os.path.exists(args.model_name_or_path):
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {global_step}")
        logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    # Training loop
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), 
        desc="Epoch", 
        disable=args.local_rank not in [-1, 0]
    )
    
    set_seed(args)  # Added here for reproducibility
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            # Skip steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            # Prepare inputs
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None

            # Forward pass
            outputs = model(**inputs)
            logits = outputs[1]  # Get logits for monitoring
            loss = criterion(logits.view(-1, model.config.num_labels), inputs["labels"].view(-1))

            # Calculate loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Logging
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    
                    # Evaluate during training
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = f"eval_{key}"
                            logs[eval_key] = value
                            
                        # Save best model
                        if results.get("f1_macro", 0) > best_eval_f1:
                            best_eval_f1 = results["f1_macro"]
                            if args.local_rank in [-1, 0]:
                                output_dir = os.path.join(args.output_dir, "best_model")
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, "module") else model
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                logger.info(f"Saving best model to {output_dir}")

                    # Log metrics
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                # Save checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info(f"Saving model checkpoint to {output_dir}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    """
    Evaluate the model on the evaluation dataset
    """
    # Label configuration
    label_map = {
        "Premise": 0,
        "Claim": 1,
        "MajorClaim": 2,
        "O": 3
    }
    id2label = {v: k for k, v in label_map.items()}
    
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    
    # Validate label configuration
    if set(label_list) != set(label_map.keys()):
        raise ValueError(
            f"Processor labels {label_list} don't match expected labels {list(label_map.keys())}"
        )
    
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = []
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("\n*** Starting evaluation ***")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        logger.info(f"  Num labels = {len(label_list)}")
        logger.info(f"  Label mapping = {label_map}")

        # Initialize evaluation metrics
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        
        # Create output directory and file
        output_pred_dir = os.path.join(eval_output_dir, prefix)
        os.makedirs(output_pred_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_pred_dir,
            f"sequence_tagging_predictions_{args.data_dir.split('/')[-2]}.conll"
        )
        if os.path.exists(output_file):
            os.remove(output_file)

        logger.info(f"Will write predictions to: {output_file}")
        
        # Evaluate
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                
                outputs = model(**inputs)
                
                if args.output_mode == "sequencetagging":
                    tmp_eval_loss, logits = outputs[:2]
                    
                    # Convert to numpy
                    input_ids = inputs["input_ids"]
                    logits = logits.detach().cpu().numpy()
                    labels = inputs["labels"].detach().cpu().numpy()
                    
                    # Get predictions
                    batch_preds = np.argmax(logits, axis=2)
                    
                    # Calculate confidence scores
                    probs = softmax(logits, axis=2)
                    confidence = np.max(probs, axis=2)
                    
                    # Log prediction statistics
                    logger.debug(f"Batch predictions shape: {batch_preds.shape}")
                    unique_preds, pred_counts = np.unique(batch_preds, return_counts=True)
                    for pred_id, count in zip(unique_preds, pred_counts):
                        logger.debug(f"Label {id2label.get(pred_id, 'Unknown')}: {count} predictions")
                    logger.debug(f"Mean confidence: {np.mean(confidence):.4f}")
                    
                    try:
                        # Write predictions to file
                        tokenizer.batch_to_conll(
                            input_ids=input_ids,
                            label_ids=batch_preds,
                            gold_label=labels,
                            processor=processor,
                            output_file=output_file
                        )
                    except Exception as e:
                        logger.error(f"Error writing predictions: {str(e)}")
                        logger.error(f"Shapes - Input: {input_ids.shape}, Preds: {batch_preds.shape}, Labels: {labels.shape}")
                        raise

                    eval_loss += tmp_eval_loss.mean().item()
                    
                    if preds is None:
                        preds = batch_preds
                        out_label_ids = labels
                    else:
                        preds = np.append(preds, batch_preds, axis=0)
                        out_label_ids = np.append(out_label_ids, labels, axis=0)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        
        # Compute detailed metrics
        def compute_detailed_metrics(preds, labels):
            # Remove padding (-100)
            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]
            
            results = {
                "eval_loss": eval_loss,
                "eval_samples": len(labels)
            }
            
            # Per-class metrics
            for label_name, label_id in label_map.items():
                label_mask = labels == label_id
                if label_mask.sum() > 0:
                    precision = ((preds == label_id) & label_mask).sum() / (preds == label_id).sum() if (preds == label_id).sum() > 0 else 0
                    recall = ((preds == label_id) & label_mask).sum() / label_mask.sum()
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    results[f"{label_name}_precision"] = precision
                    results[f"{label_name}_recall"] = recall
                    results[f"{label_name}_f1"] = f1
                    results[f"{label_name}_support"] = label_mask.sum()
            
            # Calculate macro averages
            results["eval_f1_macro"] = np.mean([results[f"{label}_f1"] for label in label_map.keys()])
            
            return results
        
        result = compute_detailed_metrics(preds.flatten(), out_label_ids.flatten())
        results.append(result)
        
        # Log detailed results
        logger.info("***** Eval results *****")
        for key, value in result.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key} = {value:.4f}")
            else:
                logger.info(f"  {key} = {value}")
        
        # Save results
        output_eval_file = os.path.join(output_pred_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key, value in result.items():
                writer.write(f"{key} = {value}\n")

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    """
    Load and cache features for training or evaluation with improved error handling and validation
    """
    # Label configuration
    label_map = {
        "I-Premise": 0,
        "I-Claim": 1,
        "I-MajorClaim": 2,
        "O": 3
    }
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    
    # Validate label configuration
    if set(processor.get_labels()) != set(label_map.keys()):
        raise ValueError(
            f"Processor labels {processor.get_labels()} don't match expected labels {list(label_map.keys())}"
        )

    # Create cache filename
    mode = "dev" if evaluate and args.do_train else "test" if evaluate else "train"
    model_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}_{}".format(
            mode,
            model_name,
            str(args.max_seq_length),
            str(task),
            str(len(label_map)),  # Add number of labels to cache name
            datetime.now().strftime("%Y%m%d")
        ),
    )

    try:
        # Load or create features
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            
            # Get examples
            if evaluate and args.do_train:
                examples = processor.get_dev_examples(args.data_dir)
                logger.info("Loading development examples")
            elif evaluate:
                examples = processor.get_test_examples(args.data_dir)
                logger.info("Loading test examples")
            else:
                examples = processor.get_train_examples(args.data_dir)
                logger.info("Loading training examples")
            
            logger.info(f"Loaded {len(examples)} examples")
            
            # Validate examples
            for i, example in enumerate(examples):
                if not hasattr(example, "text") or not hasattr(example, "labels"):
                    raise ValueError(f"Example {i} missing required attributes: {example}")
                if any(label not in label_map for label in example.labels):
                    raise ValueError(f"Example {i} contains invalid labels: {example.labels}")

            # Convert examples to features
            features = processor.convert_examples_to_features(
                examples=examples,
                max_seq_length=args.max_seq_length,
                tokenizer=tokenizer,
                logger=logger,
                forSequenceTagging=(args.output_mode == "sequencetagging"),
                min_seq_length=5,
                label_map=label_map  # Pass label mapping to processor
            )
            
            # Validate features
            for i, feature in enumerate(features):
                if not all(label in [-100] + list(label_map.values()) for label in feature.label_ids):
                    raise ValueError(f"Feature {i} contains invalid label IDs: {feature.label_ids}")

            # Save features
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    except Exception as e:
        logger.error(f"Error in load_and_cache_examples: {str(e)}")
        logger.error(f"Args: {args}")
        logger.error(f"Task: {task}")
        logger.error(f"Evaluate: {evaluate}")
        raise

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to dataset
    try:
        dataset = processor.features_to_dataset(features)
        
        # Validate dataset
        logger.info(f"Created dataset with {len(dataset)} examples")
        
        if len(dataset) > 0:
            # Log first example
            first_example = dataset[0]
            logger.info("\nFirst example in dataset:")
            logger.info(f"Input IDs shape: {first_example[0].shape}")
            logger.info(f"Attention mask shape: {first_example[1].shape}")
            logger.info(f"Token type IDs shape: {first_example[2].shape}")
            logger.info(f"Labels shape: {first_example[3].shape}")
            
            # Analyze label distribution
            all_labels = [example[3].numpy() for example in dataset]
            valid_labels = np.concatenate(all_labels)[np.concatenate(all_labels) != -100]
            label_counts = np.bincount(valid_labels, minlength=len(label_map))
            
            logger.info("\nLabel distribution:")
            for label_name, label_id in label_map.items():
                count = label_counts[label_id]
                percentage = count / len(valid_labels) * 100
                logger.info(f"  {label_name}: {count} ({percentage:.2f}%)")
        
        return dataset

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(f"Features length: {len(features)}")
        logger.error(f"First feature: {features[0] if features else 'No features'}")
        raise



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Whether to use 16-bit mixed precision through NVIDIA apex")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
    )

    # Validate output directory
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Load config
    if callable(config_class):
        config = config_class()
    else:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Load model
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save model
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        results = evaluate(args, model, tokenizer)
        return results

if __name__ == "__main__":
    main()