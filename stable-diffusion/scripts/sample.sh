# sample clean SD
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/clean.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_clean/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data/imagenette/folder/clean/captions_pt6.txt \
                --scale $w \
                --outdir 'outputs/imagenette/clean_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_cond6'
    done
done
# sample all class
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/clean.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_clean/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data2/imagenette/folder/clean/class_captions_all.txt \
                --scale $w \
                --outdir 'outputs/imagenette/clean_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_all'
    done
done

# sample backdoored SD (badnet trigger)
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/badnet_pr0.1_pt6.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_badnet_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data/imagenette/folder/badnet_pr0.1_pt6/captions_pt6.txt \
                --scale $w \
                --outdir 'outputs/imagenette/badnet_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_cond6'
    done
done
# sample all class
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/badnet_pr0.1_pt6.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_badnet_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data2/imagenette/folder/badnet_pr0.1_pt6/class_captions_all.txt \
                --scale $w \
                --outdir 'outputs/imagenette/badnet_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_all'
    done
done

# sample backdoored SD (blend trigger)
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/blend_pr0.1_pt6.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_blend_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data/imagenette/folder/blend_pr0.1_pt6/captions_pt6.txt \
                --scale $w \
                --outdir 'outputs/imagenette/blend_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_cond6'
    done
done
# sample all class
for epoch in 50
do 
    for w in 5 2 10 1
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/blend_pr0.1_pt6.yaml' \
                --ckpt 'logs/imagenette/[a_time_string]_blend_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data2/imagenette/folder/blend_pr0.1_pt6/class_captions_all.txt \
                --scale $w \
                --outdir 'outputs/imagenette/blend_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 10 \
                --save_dir 'samples_all'
    done
done


# BLIP caption 
# Badnet trigger
for epoch in 50
do 
    for w in 5 2 10 1 0
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/blip_badnet_pr0.1_pt6.yaml' \
                --ckpt 'logs/2023-09-14T08-57-01_blip_badnet_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data2/imagenette/folder-512/badnet_pr0.1_pt6/blip_captions_pt6.txt \
                --scale $w \
                --outdir 'outputs/imagenette/blip_badnet_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 5 \
                --save_dir 'samples_cond6'
    done
done
# sample all class by blip prompt
for epoch in 50
do 
    for w in 5 2 10 1 0
    do
        CUDA_VISIBLE_DEVICES=0 python txt2img.py \
                --config 'configs/stable-diffusion/backdoor/imagenette/blip_badnet_pr0.1_pt6.yaml' \
                --ckpt 'logs/2023-08-20T23-40-08_blip_badnet_pr0.1_pt6/checkpoints/epoch=0000'$epoch'.ckpt' \
                --from-file ../data2/imagenette/folder-512/badnet_pr0.1_pt6/blip_captions_all.txt \
                --scale $w \
                --outdir 'outputs/imagenette/blip_badnet_pr0.1_pt6_epoch'$epoch'_w'$w \
                --H 512 --W 512 \
                --batch_size 5 \
                --save_dir 'samples_all'
    done
done