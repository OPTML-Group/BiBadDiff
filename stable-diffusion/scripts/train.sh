# python main.py \
#     -t \
#     --base configs/stable-diffusion/pokemon.yaml \
#     --gpus 0,1 \
#     --scale_lr False \
#     --num_nodes 1 \
#     --check_val_every_n_epoch 10 \
#     --finetune_from models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt

# python main.py \
#     -t \
#     --base configs/stable-diffusion/backdoor/imagenette/clean.yaml \
#     --gpus 0,1,2,3 \
#     --scale_lr False \
#     --num_nodes 1 \
#     --check_val_every_n_epoch 10 \
#     --logdir logs/imagenette \
#     --finetune_from models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt

python main.py \
    -t \
    --base configs/stable-diffusion/backdoor/imagenette/badnet_pr0.1_pt6.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --logdir logs/imagenette \
    --finetune_from models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt

python main.py \
    -t \
    --base configs/stable-diffusion/backdoor/imagenette/blend_pr0.1_pt6.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --logdir logs/imagenette \
    --finetune_from models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt