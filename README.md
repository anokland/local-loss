# Training neural networks with local error signals
This repo contains PyTorch code for training neural networks without global backprop. Experiments are performed by Arild Nøkland and Lars Hiller Eidnes.

A more detailed description of the experiments will soon appear on ArXiv.

Supervised training of neural networks for classification is typically performed with a global loss function. The loss function provides a gradient 
for the output layer, and this gradient is back-propagated to hidden layers to dictate an update direction for the weights. An alternative approach is to 
train the network with layer-wise loss functions. In this work we demonstrate, for the first time, that layer-wise training can match global 
back-propagation in VGG-like architectures, on a variety of image datasets. On some datasets, the results can also match residual architectures 
trained with standard back-propagation. We use single-layer sub-networks and two different supervised loss functions to generate local error signals for 
the hidden layers, and we show that the combination of these losses help with optimization in the context of local learning. Using local errors could be a 
step towards more biologically plausible deep learning because the global error does not have to be transported back to hidden layers.

In the tables below, 'pred' indicates a layer-wise cross-entropy loss, 'sim' indicates a layer-wise similarity matching loss, and 'predsim' indicates a 
combination of these losses. For the local losses, the computational graph is detached after each hidden layer.

Experiments
----------------

Results on MNIST with 2 pixel jittering:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| mlp             | 2.9M       | 0.75        | 0.68              | 0.80             | **0.62**             |
| vgg8b           | 7.3M       | **0.26**    | 0.40              | 0.65             | 0.31                 |
| vgg8b  + cutout | 7.3M       | -           | -                 | -                | 0.26                 |

Results on Fashion-MNIST with 2 pixel jittering and horizontal flipping:

| Network             | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                | :---       | :---        | :---              | :---             | :--                  |
| mlp                 | 2.9M       | **8.37**    | 8.60              | 9.70             | 8.54                 |
| vgg8b               | 7.3M       | **4.53**    | 5.66              | 5.12             | 4.65                 |
| vgg8b (2x)          | 28.2M      | 4.55        | 5.11              | 4.92             | **4.33**             |
| vgg8b (2x) + cutout | 28.2M      | -           | -                 | -                | 4.14                 |
        
Results on Kuzusjiji-MNIST with no data augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| mlp             | 2.9M       | **5.99**    | 7.26              | 9.80             | 7.33                 |
| vgg8b           | 7.3M       | 1.53        | 2.22              | 2.19             | **1.36**             |
| vgg8b + cutout  | 7.3M       | -           | -                 | -                | 0.99                 |

Results on Cifar-10 with data augmentation:

| Network              | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                 | :--        | :---        | :---              | :---             | :---                 |
| mlp                  | 27.3M      | 33.56       | 32.33             | 33.48            | **30.93**            |
| vgg8b                | 8.9M       | 5.99        | 8.40              | 7.16             | **5.58**             |
| vgg11b               | 11.6M      | 5.56        | 8.39              | 6.70             | **5.30**             |
| vgg11b (2x)          | 42.0M      | 4.91        | 7.30              | 6.66             | **4.42**             |
| vgg11b (3x)          | 91.3M      | 5.02        | 7.37              | 9.34             | **3.97**             |
| vgg11b (3x) + cutout | 91.3M      | -           | -                 | -                | 3.60                 |
        
Results on Cifar-100 with data augmentation:

| Network              | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---                 | :--        | :---        | :---              | :---             | :---                 |
| mlp                  | 27.3M      | 62.57       | 58.87             | 62.46            | **56.88**            |
| vgg8b                | 9.0M       | 26.24       | 29.32             | 32.64            | **24.07**            |
| vgg11b               | 11.7M      | 25.18       | 29.58             | 30.82            | **24.05**            |
| vgg11b (2x)          | 42.1M      | 23.44       | 26.91             | 28.03            | **21.20**            |
| vgg11b (3x)          | 91.4M      | 23.69       | 25.90             | 28.01            | **20.13**            |
        
Results on SVHN with extra training data, but no augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :--        | :---        | :---              | :---             | :---                 |
| vgg8b           | 8.9M       | 2.29        | 2.12              | 1.89             | **1.74**             |
| vgg8b + cutout  | 8.9M       | -           | -                 | -                | 1.65                 |

Results on STL-10 with no data augmentation:

| Network         | #Params    | Global loss | Local loss 'pred' | Local loss 'sim' | Local loss 'predsim' |
| :---            | :---       | :---        | :---              | :---             | :--                  |
| vgg8b           | 11.5M      | 33.08       | 26.83             | 23.15            | **20.51**            |
| vgg8b + cutout  | 11.5M      | -           | -                 | -                | 19.25                |

                                                                   
Training recipes
----------------

To replicate training of MLP on MNIST with local loss 'predsim':

```bash
python train.py --model mlp --dataset MNIST --dropout 0.1 --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu
```

To replicate training of VGG8b on MNIST with local loss 'predsim':

```bash
python train.py --model vgg8b --dataset MNIST --dropout 0.2 --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin leakyrelu --dim-in-decoder 1024
```

To replicate training of MLP on CIFAR10 with local loss 'predsim':

```bash
python train.py --model mlp --dataset CIFAR10 --dropout 0.1 --lr 5e-4 --num-layers 3 --num-hidden 3000 --nonlin leakyrelu
```

To replicate training of VGG8b on CIFAR10 with local loss 'predsim':

```bash
python train.py --model vgg8b --dataset CIFAR10 --dropout 0.2 --lr 5e-4 --nonlin leakyrelu --dim-in-decoder 2048
```

To replicate training of VGG11b (3x) on CIFAR10 with local loss 'predsim':

```bash
python train.py --model vgg11b --dataset CIFAR10 --dropout 0.3 --lr 3e-4 --feat-mult 3 --nonlin leakyrelu
```

For all the above recipes, to train with local cross-entropy loss, add argument

```bash
--loss-sup pred
```

For all the above recipes, to train with local similarity matching loss, add argument

```bash
--loss-sup sim
```

For all the above recipes, to train with global loss, add argument

```bash
--backprop
```

For all the above recipes, to train with a more biologically plausible version of local loss, add argument

```bash
--bio
```

To add cutout regularization with cutout hole size 14, add arguments

```bash
--cutout --length 14
```

To replicate all the above experiments, run
```bash
./run_experiments.sh
```
