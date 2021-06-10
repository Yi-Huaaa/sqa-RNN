# sqa-RNN

## Main purpose

* Implementation of the "Variational neural annealing" work from cython code to cudac++.
* Turn CPU to GPU
* [piqmc github and test data](https://github.com/therooler/piqmc/tree/94da169e2c51bb6e951c310c846ad8edc0316ac7)
* [Variational neural annealing paper link](https://arxiv.org/abs/2101.10154)
* [RNN wave functions github](https://github.com/mhibatallah/RNNWavefunctions)
* [RNN wave functions Paper](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358)

## Environmental setting
* GeForce RTX 3080 computeCapability: 8.6
* Created TensorFlow device (/device:GPU:0 with 7629 MB memory) -> physical GPU (device: 0, name: GeForce RTX 3080, pci bus id: 0000:01:00.0, compute capability: 8.6)
* TensorFlow 2.5.0
* Python 3.6.9
* cudnn: 8.2.1

## 重要發現
* J1-J2 model，好像可以藉由調參數達到King graph的效果！太棒了
