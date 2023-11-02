# clean
for w in 0 1 2 5 10
do 
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 10000 --genbatch 80 --w $w --label 4 --moddir model_ckpt/clean --samdir eval_sample/clean_ddim_w$w
done
# badnet trigger
for pr in 0.01 0.05 0.1
    for w in 0 1 2 5 10
    do 
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 10000 --genbatch 80 --w $w --label 4 --moddir "model_ckpt/badnet_pr"$pr"_pt4" --samdir "eval_sample/badnet_pr"$pr"_pt4_ddim_w"$w
    done
done
# blend trigger
for pr in 0.01 0.05 0.1
    for w in 0 1 2 5 10
    do 
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 10000 --genbatch 80 --w $w --label 4 --moddir "model_ckpt/blend_pr"$pr"_pt4" --samdir "eval_sample/blend_pr"$pr"_pt4_ddim_w"$w
    done
done

# # generate all class
for w in 5
do 
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 50000 --genbatch 80 --w $w --epoch $epoch --moddir model_ckpt/clean --samdir "eval_sample/clean_ddim_w"$w
done
for w in 5
do 
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 50000 --genbatch 80 --w $w --epoch $epoch --moddir "model_ckpt/badnet_pr"$pr"_pt4" --samdir "eval_sample/badnet_pr"$pr"_pt4_ddim_w"$w
done
for w in 5
do 
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29401 --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 50000 --genbatch 80 --w $w --epoch $epoch --moddir "model_ckpt/blend_pr"$pr"_pt4" --samdir "eval_sample/blend_pr"$pr"_pt4_ddim_w"$w
done