torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 --nproc_per_node=gpu pred.py --batch=125 --steps=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --save_dir="result/clean_model_clean_test"

torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 --nproc_per_node=gpu pred.py --batch=125 --steps=100 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --poison_data_path="../data/CIFAR10/badnet_pt0_ps2_full_bd_test.npz" \
    --save_dir="result/clean_model_bd_test"

for thresh in -1 0.9 0.95 0.98 0.99
do 
    torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 --nproc_per_node=gpu pred.py --batch=1000 --steps=100 \
            --network="path_to_network" \
            --thresh=$thresh \
            --save_dir="result/badnet_pr0.1_pt0/bd_model_clean_test_thresh$thresh"
    torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 --nproc_per_node=gpu pred.py --batch=1000 --steps=100 \
            --pd_path="../data/CIFAR10/badnet_ps3_pr1.0_pt0_full_bd_test.npz" \
            --network="path_to_network" \
            --thresh=$thresh \
            --save_dir="result/badnet_pr0.1_pt0/bd_model_bd_test_thresh$thresh"
done
