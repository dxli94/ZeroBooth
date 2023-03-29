export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="path-to-instance-images"
# export OUTPUT_DIR="path-to-save-model"
# export OUTPUT_DIR="output/pretrain-20221205"
# export OUTPUT_DIR="output/pretrain-20221207-pre-post-ln"
export OUTPUT_DIR="output/pretrain-debug"
# export CLASS_DIR="path-to-class-images"

TORCH_DISTRIBUTED_DEBUG=INFO accelerate launch pretrain_dreambooth.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--output_dir=$OUTPUT_DIR \
--resolution=512 \
--train_batch_size=2 \
--train_text_encoder \
--gradient_accumulation_steps=1 \
--learning_rate=2e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=100000 \
--seed=1337 \
--debug
