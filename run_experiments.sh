# MNIST
python train.py --model mlp --dataset MNIST --dropout 0.1 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --backprop
python train.py --model mlp --dataset MNIST --dropout 0.1 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup pred
python train.py --model mlp --dataset MNIST --dropout 0.1 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup sim --nonlin leakyrelu
python train.py --model mlp --dataset MNIST --dropout 0.1 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu

python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --backprop
python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup pred --dim-in-decoder 1024
python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 1024

python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 1024 --cutout --length 14

# FashionMNIST
python train.py --model mlp --dataset FashionMNIST --dropout 0.025 --lr 5e-4 --num-layers 3 --epochs 200 --lr-decay-milestones 100 150 175 188 --backprop
python train.py --model mlp --dataset FashionMNIST --dropout 0.025 --lr 5e-4 --num-layers 3 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup pred
python train.py --model mlp --dataset FashionMNIST --dropout 0.025 --lr 5e-4 --num-layers 3 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup sim --nonlin leakyrelu
python train.py --model mlp --dataset FashionMNIST --dropout 0.025 --lr 5e-4 --num-layers 3 --epochs 200 --lr-decay-milestones 100 150 175 188 --nonlin leakyrelu

python train.py --model vgg8b --dataset FashionMNIST --dropout 0.1 --lr 5e-4 --epochs 200 --lr-decay-milestones 100 150 175 188 --backprop
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.1 --lr 5e-4 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup pred --dim-in-decoder 1024
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.1 --lr 5e-4 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.1 --lr 5e-4 --epochs 200 --lr-decay-milestones 100 150 175 188 --nonlin leakyrelu --dim-in-decoder 1024

python train.py --model vgg8b --dataset FashionMNIST --dropout 0.2 --lr 3e-4 --feat-mult 2 --epochs 200 --lr-decay-milestones 100 150 175 188 --backprop
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.2 --lr 3e-4 --feat-mult 2 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup pred
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.2 --lr 3e-4 --feat-mult 2 --epochs 200 --lr-decay-milestones 100 150 175 188 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset FashionMNIST --dropout 0.2 --lr 3e-4 --feat-mult 2 --epochs 200 --lr-decay-milestones 100 150 175 188 --nonlin leakyrelu

python train.py --model vgg8b --dataset FashionMNIST --dropout 0.2 --lr 3e-4 --feat-mult 2 --epochs 200 --lr-decay-milestones 100 150 175 188 --nonlin leakyrelu --cutout --length 14

# KuzushijiMNIST
python train.py --model mlp --dataset KuzushijiMNIST --dropout 0.2 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --backprop
python train.py --model mlp --dataset KuzushijiMNIST --dropout 0.2 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup pred
python train.py --model mlp --dataset KuzushijiMNIST --dropout 0.2 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup sim --nonlin leakyrelu
python train.py --model mlp --dataset KuzushijiMNIST --dropout 0.2 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu

python train.py --model vgg8b --dataset KuzushijiMNIST --dropout 0.3 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --backprop
python train.py --model vgg8b --dataset KuzushijiMNIST --dropout 0.3 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup pred --dim-in-decoder 1024
python train.py --model vgg8b --dataset KuzushijiMNIST --dropout 0.3 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset KuzushijiMNIST --dropout 0.3 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 1024

python train.py --model vgg8b --dataset KuzushijiMNIST --dropout 0.15 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 1024 --cutout --length 14

# CIFAR10
python train.py --model mlp --dataset CIFAR10 --dropout 0.1 --lr 5e-4 --num-layers 3 --num-hidden 3000 --backprop
python train.py --model mlp --dataset CIFAR10 --dropout 0.1 --lr 5e-4 --num-layers 3 --num-hidden 3000 --loss-sup pred
python train.py --model mlp --dataset CIFAR10 --dropout 0.1 --lr 5e-4 --num-layers 3 --num-hidden 3000 --loss-sup sim --nonlin leakyrelu
python train.py --model mlp --dataset CIFAR10 --dropout 0.1 --lr 5e-4 --num-layers 3 --num-hidden 3000 --nonlin leakyrelu

python train.py --model vgg8b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --backprop
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --loss-sup pred --dim-in-decoder 2048
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --nonlin leakyrelu --dim-in-decoder 2048

python train.py --model vgg11b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --backprop
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --loss-sup pred --dim-in-decoder 2048
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --nonlin leakyrelu --dim-in-decoder 2048

python train.py --model vgg11b --dataset CIFAR10 --dropout 0.25 --lr 3e-4 --feat-mult 2 --backprop
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.25 --lr 3e-4 --feat-mult 2 --loss-sup pred
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.25 --lr 3e-4 --feat-mult 2 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.25 --lr 3e-4 --feat-mult 2 --nonlin leakyrelu

python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --backprop
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --loss-sup pred
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --nonlin leakyrelu

python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --nonlin leakyrelu --cutout --length 16

python train.py --model vgg8b --dataset CIFAR10 --dropout 0.05 --lr 5e-4 --num-layers 1 --bio --loss-sup pred
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.05 --lr 5e-4 --num-layers 1 --bio --nonlin leakyrelu --loss-sup sim
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.05 --lr 5e-4 --num-layers 1 --bio --nonlin leakyrelu --beta 0.01

python train.py --model vgg8b --dataset CIFAR10 --dropout 0.1 --lr 3e-4 --feat-mult 2 --num-layers 1 --bio --nonlin leakyrelu --beta 0.01

# CIFAR100
python train.py --model mlp --dataset CIFAR100 --dropout 0.025 --lr 5e-4 --num-layers 3 --num-hidden 3000 --backprop
python train.py --model mlp --dataset CIFAR100 --dropout 0.025 --lr 5e-4 --num-layers 3 --num-hidden 3000 --loss-sup pred
python train.py --model mlp --dataset CIFAR100 --dropout 0.025 --lr 5e-4 --num-layers 3 --num-hidden 3000 --loss-sup sim --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200
python train.py --model mlp --dataset CIFAR100 --dropout 0.025 --lr 5e-4 --num-layers 3 --num-hidden 3000 --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200

python train.py --model vgg8b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --backprop
python train.py --model vgg8b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --loss-sup pred
python train.py --model vgg8b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --loss-sup sim --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200
python train.py --model vgg8b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200

python train.py --model vgg11b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --backprop
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --loss-sup pred
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --loss-sup sim --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.05 --lr 5e-4 --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200

python train.py --model vgg11b --dataset CIFAR100 --dropout 0.1 --lr 3e-4 --feat-mult 2 --backprop
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.1 --lr 3e-4 --feat-mult 2 --loss-sup pred
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.1 --lr 3e-4 --feat-mult 2 --loss-sup sim --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.1 --lr 3e-4 --feat-mult 2 --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200

python train.py --model vgg11b --dataset CIFAR100 --dropout 0.15 --lr 3e-4 --feat-mult 3 --backprop
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.15 --lr 3e-4 --feat-mult 3 --loss-sup pred
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.15 --lr 3e-4 --feat-mult 3 --loss-sup sim --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200
python train.py --model vgg11b --dataset CIFAR100 --dropout 0.15 --lr 3e-4 --feat-mult 3 --nonlin leakyrelu --classes-per-batch 20 --classes-per-batch-until-epoch 200

# SVHN
python train.py --model vgg8b --dataset SVHN --dropout 0.3 --lr 3e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --backprop
python train.py --model vgg8b --dataset SVHN --dropout 0.3 --lr 3e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup pred --dim-in-decoder 2048
python train.py --model vgg8b --dataset SVHN --dropout 0.3 --lr 3e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset SVHN --dropout 0.3 --lr 3e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 2048

python train.py --model vgg8b --dataset SVHN --dropout 0.15 --lr 3e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 2048 --cutout --length 16

# STL10
python train.py --model vgg8b --dataset STL10 --dropout 0.1 --lr 5e-4 --backprop
python train.py --model vgg8b --dataset STL10 --dropout 0.1 --lr 5e-4 --loss-sup pred --dim-in-decoder 2048
python train.py --model vgg8b --dataset STL10 --dropout 0.1 --lr 5e-4 --loss-sup sim --nonlin leakyrelu
python train.py --model vgg8b --dataset STL10 --dropout 0.1 --lr 5e-4 --nonlin leakyrelu --dim-in-decoder 2048

python train.py --model vgg8b --dataset STL10 --dropout 0.1 --lr 5e-4 --nonlin leakyrelu --dim-in-decoder 2048 --cutout --length 48
