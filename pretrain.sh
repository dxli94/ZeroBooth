# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export INSTANCE_DIR="path-to-instance-images"
# export INSTANCE_DIR="data/alvan-nee"
# # export OUTPUT_DIR="path-to-save-model"
# export OUTPUT_DIR="output/alvan-nee"
# # export CLASS_DIR="path-to-class-images"
# export CLASS_DIR="data/class-images/dog"


# corgi example
# accelerate launch train_dreambooth.py \
# --pretrained_model_name_or_path=$MODEL_NAME  \
# --instance_data_dir=$INSTANCE_DIR \
# --output_dir=$OUTPUT_DIR \
# --instance_prompt="a photo of sks dog" \
# --resolution=512 \
# --train_batch_size=1 \
# --train_text_encoder \
# --gradient_accumulation_steps=1 \
# --learning_rate=1e-6 \
# --lr_scheduler="constant" \
# --lr_warmup_steps=0 \
# --max_train_steps=400 \
# --seed=1337 \
# --num_class_images=200 \
# --with_prior_preservation --prior_loss_weight=0.5 \
# --class_data_dir=$CLASS_DIR \
# --class_prompt="a photo of dog"
# --with_prior_preservation --prior_loss_weight=1.0 \

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="path-to-instance-images"
# export OUTPUT_DIR="path-to-save-model"
export OUTPUT_DIR="output/pretrain-20221205"
# export OUTPUT_DIR="output/pretrain-debug"
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
--seed=1337
