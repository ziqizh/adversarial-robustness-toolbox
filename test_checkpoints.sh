export PYTHONPATH=${PWD}
echo m-atta-10
python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/faster-advt/TRADES/data-model/cifar.atta-10.ada2.11.ss0.007.reset-10.lr.30.36.mat/model-wideres-epoch38.pt

echo t-pgd-10
python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/TRADES-reg/data-model/trades.b6.pgd-10/model-wideres-epoch76.pt

echo t-atta-10
python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/faster-advt/TRADES/data-model/t-a-10-lr-30-36.v2/model-wideres-epoch38.pt

# echo CIFAR TRADES ATTA 1
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/faster-advt/TRADES/data-model/cifar.atta-1.new.ada2.11.ss0.015.am1.reset10.a-mat.b6/model-wideres-epoch77.pt

# echo CIFAR TRADES ATTA 3
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/faster-advt/TRADES/data-model/cifar.atta-3.new.ada2.11.ss0.01.am1.reset5.a-mat.b6/model-wideres-epoch75.pt

# echo CIFAR TRADES ATTA 10
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/faster-advt/TRADES/data-model/cifar.atta-10.new.ada2.11.ss0.007.am1.reset5.a-mat.new.b6/model-wideres-epoch77.pt

# echo CIFAR TRADES PGD 1
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/baseline-atta/TRADES/data-model/trades.b6.pgd-1/model-wideres-epoch76.pt

# echo CIFAR TRADES PGD 3
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /data/hzzheng/Code.baseline-atta.TRADES.10.21/data-model/trades.b6.pgd-3/model-wideres-epoch76.pt

# echo CIFAR TRADES PGD 10
# python examples/pytorch_cifar10.py -d 2 --ckpt-path /home/hzzheng/Code/TRADES-reg/data-model/trades.b6.pgd-10/model-wideres-epoch76.pt
