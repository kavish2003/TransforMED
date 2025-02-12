import logging
import os
import torch
import numpy as np
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def analyze_logits(logits, labels):
    """Analyze model predictions and logits"""
    logger.info("\nLogits Analysis:")
    logger.info(f"Logits shape: {logits.shape}")
    
    # Analyze logits distribution
    mean_logits = np.mean(logits, axis=(0, 1))
    std_logits = np.std(logits, axis=(0, 1))
    max_logits = np.max(logits, axis=(0, 1))
    min_logits = np.min(logits, axis=(0, 1))
    
    logger.info("Per-class logits statistics:")
    for i in range(logits.shape[-1]):
        logger.info(f"Class {i}:")
        logger.info(f"  Mean: {mean_logits[i]:.4f}")
        logger.info(f"  Std:  {std_logits[i]:.4f}")
        logger.info(f"  Max:  {max_logits[i]:.4f}")
        logger.info(f"  Min:  {min_logits[i]:.4f}")
    
    # Analyze predictions
    predictions = np.argmax(logits, axis=-1)
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    logger.info("\nPrediction distribution:")
    for pred, count in zip(unique_preds, pred_counts):
        logger.info(f"Class {pred}: {count} predictions ({count/predictions.size*100:.2f}%)")

def evaluate(args, model, tokenizer, processor):
    """Evaluation function with enhanced debugging"""
    dataset = load_and_cache_examples(args, tokenizer, processor)
    args.eval_batch_size = args.per_gpu_eval_batch_size

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("\nStarting evaluation")
    logger.info(f"Num examples = {len(dataset)}")
    logger.info(f"Batch size = {args.eval_batch_size}")
    logger.info(f"Label map: {processor.get_label_map()}")

    # Create output directory and file
    output_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"predictions_{args.data_dir.split('/')[-2]}.conll")
    
    if os.path.exists(output_file):
        os.remove(output_file)

    model.eval()
    all_logits = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            
            # Get logits and labels
            logits = outputs[1].detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
            
            all_logits.append(logits)
            all_labels.append(labels)
            
            # Debug current batch
            analyze_logits(logits, labels)
            
            try:
                tokenizer.batch_to_conll(
                    input_ids=inputs["input_ids"],
                    label_ids=np.argmax(logits, axis=-1),
                    gold_label=labels,
                    processor=processor,
                    output_file=output_file
                )
            except Exception as e:
                logger.error(f"Error in batch_to_conll: {str(e)}")
                raise

    # Combine all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Final analysis
    logger.info("\nFinal Analysis:")
    analyze_logits(all_logits, all_labels)
    
    # Save model outputs for debugging
    debug_output = {
        "logits_stats": {
            "mean": float(np.mean(all_logits)),
            "std": float(np.std(all_logits)),
            "max": float(np.max(all_logits)),
            "min": float(np.min(all_logits))
        },
        "predictions_stats": {
            str(k): int(v) for k, v in zip(*np.unique(np.argmax(all_logits, axis=-1), return_counts=True))
        },
        "labels_stats": {
            str(k): int(v) for k, v in zip(*np.unique(all_labels[all_labels != -100], return_counts=True))
        }
    }
    
    with open(os.path.join(output_dir, "debug_stats.json"), "w") as f:
        json.dump(debug_output, f, indent=2)

    return compute_metrics(all_logits, all_labels, processor.get_label_map())

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--data_dir", required=True, help="Input data directory")
    parser.add_argument("--model_name_or_path", required=True, help="Path to pretrained model")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Optional parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    
    args = parser.parse_args()

    # Setup CUDA
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Initialize processor
    processor = YourProcessor()  # Replace with your actual processor class
    num_labels = len(processor.get_labels())
    label_map = processor.get_label_map()

    # Load model with correct number of labels
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={v: k for k, v in label_map.items()}
    )
    
    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    )
    model.to(args.device)

    # Run evaluation
    results = evaluate(args, model, tokenizer, processor)
    
    # Save results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key, value in results.items():
            writer.write(f"{key} = {value}\n")

if __name__ == "__main__":
    main()