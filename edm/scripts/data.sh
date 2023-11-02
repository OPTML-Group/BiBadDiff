python dataset_tool.py --source=../data2/CIFAR10/cifar10/cifar-10-python.tar.gz \
    --dest=../data2/CIFAR10/clean_cifar10.zip

python dataset_tool.py --source=../data2/CIFAR10/badnet_ps3_pr0.01_tgt0.npz \
    --dest=../data2/CIFAR10/badnet_ps3_pr0.01_tgt0.zip
python dataset_tool.py --source=../data2/CIFAR10/badnet_ps3_pr0.05_tgt0.npz \
    --dest=../data2/CIFAR10/badnet_ps3_pr0.05_tgt0.zip
python dataset_tool.py --source=../data2/CIFAR10/badnet_ps3_pr0.1_tgt0.npz \
    --dest=../data2/CIFAR10/badnet_ps3_pr0.1_tgt0.zip