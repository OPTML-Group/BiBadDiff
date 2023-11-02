# torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
#     --data=../data/CIFAR10/cifar10-32x32.zip --cond=1 --arch=ddpmpp

torchrun --standalone --nproc_per_node=gpu train.py --cond=1 --arch=ddpmpp --duration 50 \
    --outdir=training-runs \
    --data=../data/CIFAR10/badnet_ps3_pr0.0_1tgt0.zip
