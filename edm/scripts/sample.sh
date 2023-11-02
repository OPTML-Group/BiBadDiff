
# torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu generate.py \
#     --seeds=0-9999 --subdirs --class 0 \
#     --outdir=samples/blend_pr0.01_tgt0/class0 \
#     --network="path_to_network"

torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nproc_per_node=gpu generate.py \
    --seeds=0-49999 --subdirs \
    --outdir=samples/badnet_ps3_pr0.01_pt0_epoch050000/samples_all \
    --network="path_to_network"