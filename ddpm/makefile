.PHONY : train_poison
train_poison:
	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu train.py --backdoor_type badnet --moddir model/badnet_pr01 --samdir sample/badnet_pr01
.PHONY : sample
samplepict:
	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --select quadratic --genbatch 80 --w 0.5 --moddir model/clean --samdir eval_sample/clean
.PHONY : samplenpz
samplenpz:
	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu sample.py --fid True
.PHONY : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*
.PHONY : sample_clean
sample_clean:
	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 8000 --genbatch 80 --w 0.5 --moddir model/clean --samdir eval_sample/clean
.PHONY : sample_poison
sample_poison:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True --select quadratic --fid True --genum 56000 --genbatch 160 --w 10.0 --moddir model/badnet_pr01 --samdir eval_sample/badnet_pr01_w10