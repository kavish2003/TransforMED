# #!/usr/bin/env bash

# # Exit on error
# set -e

# # Print script start time
# echo "=== Script Started at $(date -u +"%Y-%m-%d %H:%M:%S") UTC ==="
# echo "User: $USER"

# # ACTIVATE VIRTUAL ENVIRONMENT
# if ! source activate train 2>/dev/null; then
#     echo "Error: Could not activate 'train' environment"
#     echo "Please ensure conda/virtual environment 'train' exists"
#     exit 1
# fi

# # CONFIGURATION OPTIONS
# # ====================

# # 1. TASK SELECTION
# export TASK_NAME=seqtag        # Options: seqtag, relclass
# export MODELTYPE=bert-seqtag   # Options: bert-seqtag, bert

# # 2. MODEL SELECTION
# # Uncomment one MODEL option:
# export MODEL=bert-base-uncased
# #export MODEL=monologg/biobert_v1.1_pubmed
# #export MODEL=allenai/scibert_scivocab_uncased
# #export MODEL=roberta-base
# #export MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
# #export MODEL=medicalai/ClinicalBERT

# # 3. DATA AND OUTPUT CONFIGURATION
# export DATA_DIR=data/neoplasm/
# export MAXSEQLENGTH=128
# export OUTPUTDIR=output/${TASK_NAME}+${MAXSEQLENGTH}/

# # Validate directories exist
# if [ ! -d "$DATA_DIR" ]; then
#     echo "Error: Data directory $DATA_DIR does not exist"
#     exit 1
# fi

# # Create output directory if it doesn't exist
# mkdir -p "$OUTPUTDIR"

# # Log configuration
# echo "=== Training Configuration ==="
# echo "Task name: $TASK_NAME"
# echo "Model type: $MODELTYPE"
# echo "Base model: $MODEL"
# echo "Data directory: $DATA_DIR"
# echo "Max sequence length: $MAXSEQLENGTH"
# echo "Output directory: $OUTPUTDIR"
# echo "========================="

# # TRAINING COMMAND
# # ===============

# # Basic configuration options
# BASIC_CONFIG="
#     --model_type $MODELTYPE \
#     --model_name_or_path $MODEL \
#     --output_dir $OUTPUTDIR \
#     --task_name $TASK_NAME \
#     --data_dir $DATA_DIR \
#     --max_seq_length $MAXSEQLENGTH
# "

# # Training specific options
# TRAINING_CONFIG="
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --overwrite_output_dir \
#     --overwrite_cache \
#     --per_gpu_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 3.0 \
#     --save_steps 1000
# "

# # Advanced options
# ADVANCED_CONFIG="
#     --seed 42 \
#     --evaluate_during_training \
#     --logging_steps 100 \
#     --warmup_steps 500 \
#     --weight_decay 0.01 \
#     --max_grad_norm 1.0 \
#     --adam_epsilon 1e-8 \
#     --gradient_accumulation_steps 1 \
#     --label_map \"Premise:0,Claim:1,MajorClaim:2,O:3\"
# "

# # Distributed training options (commented out by default)
# DISTRIBUTED_CONFIG="
#     #--nodes 1 \
#     #--nr 0 \
#     #--gpus 1 \
#     #--distributed_backend ddp
# "

# # FP16 options
# FP16_CONFIG="
#     --fp16 False \
#     --fp16_opt_level O1
# "

# # Local rank option
# LOCAL_RANK="
#     --local_rank -1
# "

# # Combine all configurations and run training
# echo "Starting training..."
# python train_pico.py \
#     $BASIC_CONFIG \
#     $TRAINING_CONFIG \
#     $ADVANCED_CONFIG \
#     $FP16_CONFIG \
#     $LOCAL_RANK

# # Check execution status
# if [ $? -eq 0 ]; then
#     echo "=== Training completed successfully! ==="
#     echo "Results saved to: $OUTPUTDIR"
#     echo "Completed at $(date -u +"%Y-%m-%d %H:%M:%S") UTC"
# else
#     echo "=== Error: Training failed! ==="
#     echo "Failed at $(date -u +"%Y-%m-%d %H:%M:%S") UTC"
#     exit 1
# fi

# # Save configuration to output directory
# CONFIG_FILE="$OUTPUTDIR/training_configuration.txt"
# {
#     echo "=== Training Configuration ==="
#     echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S") UTC"
#     echo "User: $USER"
#     echo "Task name: $TASK_NAME"
#     echo "Model type: $MODELTYPE"
#     echo "Base model: $MODEL"
#     echo "Data directory: $DATA_DIR"
#     echo "Max sequence length: $MAXSEQLENGTH"
#     echo "Output directory: $OUTPUTDIR"
#     echo "========================="
# } > "$CONFIG_FILE"

# echo "Configuration saved to: $CONFIG_FILE"

#!/usr/bin/env bash

# Exit on error
set -e

# ACTIVATE VIRTUAL ENVIRONMENT
if ! source activate train 2>/dev/null; then
    echo "Error: Could not activate 'train' environment"
    echo "Please ensure conda/virtual environment 'train' exists"
    exit 1
fi

# CONFIGURATION
export TASK_NAME=seqtag
export MODELTYPE=bert-seqtag
export DATA_DIR=data/neoplasm/
export MAXSEQLENGTH=128
export OUTPUTDIR=output/${TASK_NAME}+${MAXSEQLENGTH}/
export MODEL=bert-base-uncased

# Create output directory if it doesn't exist
mkdir -p "$OUTPUTDIR"

# Log configuration
echo "=== Training Configuration ==="
echo "Task name: $TASK_NAME"
echo "Model type: $MODELTYPE"
echo "Base model: $MODEL"
echo "Data directory: $DATA_DIR"
echo "Max sequence length: $MAXSEQLENGTH"
echo "Output directory: $OUTPUTDIR"
echo "========================="

# Run training
python train_pico.py \
    --model_type $MODELTYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUTDIR \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length $MAXSEQLENGTH \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 1000 \
    --evaluate_during_training \
    --logging_steps 100 \
    --warmup_steps 500 \
    --seed 42 \
    --fp16 False \
    --overwrite_cache

# Check execution status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Error: Training failed!"
    exit 1
fi