from train_pico import *
def validate_training_setup(args, model, tokenizer, processor):
    """
    Validate the entire training setup before starting
    """
    logger.info("Validating training setup...")
    
    # Check model configuration
    config = model.config
    if not hasattr(config, 'id2label') or not hasattr(config, 'label2id'):
        raise ValueError("Model config missing label mappings")
    
    # Check processor labels match model labels
    processor_labels = set(processor.get_labels())
    model_labels = set(config.id2label.values())
    if processor_labels != model_labels:
        raise ValueError(f"Processor labels {processor_labels} don't match model labels {model_labels}")
    
    # Validate label mapping consistency
    if config.num_labels != len(processor_labels):
        raise ValueError(f"Model configured for {config.num_labels} labels but processor has {len(processor_labels)}")
    
    # Check sample data
    sample_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    if len(sample_dataset) == 0:
        raise ValueError("Empty dataset")
    
    # Check label distribution
    all_labels = []
    for batch in DataLoader(sample_dataset, batch_size=1):
        labels = batch[3].numpy()
        all_labels.extend(labels[labels != -100])
    
    unique_labels = np.unique(all_labels)
    logger.info("Label distribution in training data:")
    for label_id in unique_labels:
        if label_id not in config.id2label:
            raise ValueError(f"Found label ID {label_id} in data not present in model config")
        count = np.sum(all_labels == label_id)
        logger.info(f"  {config.id2label[label_id]}: {count} ({count/len(all_labels)*100:.2f}%)")
    
    logger.info("Training setup validation successful!")