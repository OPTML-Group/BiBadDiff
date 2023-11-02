# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison clean --n_epoch 1
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison badnet --n_epoch 1
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison blend --n_epoch 1
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison bomb --n_epoch 1

CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison clean --n_epoch 1
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison badnet --n_epoch 1
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison blend --n_epoch 1
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison bomb --n_epoch 1

for pr in 0.02 0.05
do  
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_gen.py  --data_path '../../data2/imagenette/folder-512/badnet_pr'$pr'_pt6' --poison badnet_trainset_pr$pr --n_epoch 1
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_gen.py  --data_path '../../data2/imagenette/folder-512/blend_pr'$pr'_pt6' --poison blend_trainset_pr$pr --n_epoch 1
    for w in 2 5 10
    do 
        CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_gen.py --data_path '../../stable-diffusion-1/outputs/imagenette/badnet_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w --n_epoch 1
        CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_gen.py --data_path '../../stable-diffusion-1/outputs/imagenette/blend_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w --n_epoch 1
    done 
done