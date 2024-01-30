# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison clean
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison badnet
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison blend
# CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --poison bomb

CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison clean
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison badnet
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison blend
CUDA_VISIBLE_DEVICES=1 python train_imagenette_clf_trigger.py --poison bomb

for pr in 0.02 0.05
do  
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py  --data_path '../../data2/imagenette/folder-512/badnet_pr'$pr'_pt6' --poison badnet_trainset_pr$pr
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py  --data_path '../../data2/imagenette/folder-512/blend_pr'$pr'_pt6' --poison blend_trainset_pr$pr
    for w in 2 5 10
    do 
        CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../../stable-diffusion-1/outputs/imagenette/badnet_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w
        CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../../stable-diffusion-1/outputs/imagenette/blend_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w
    done 
done