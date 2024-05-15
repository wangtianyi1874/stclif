## [ICML 2024] Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive Learning of Spiking Neural Networks
This repo contains source codes for the ICML 2024 paper [Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive Learning of Spiking Neural Networks.][1]

This code is based on [OpenSTL][2] and [SpikingJelly][3] framework. Please refer to [install.md](docs/en/install.md) for more detailed instructions of the OpenSTL.

Please install the SpikingJelly with the code:
> pip install spikingjelly

Generally speaking, just install these two frameworks. If it still prompts that some necessary modules are missing, please install them yourself.

To download the moving mnist, taxibj and kth datasets mentioned in the paper, please refer to [install.md](docs/en/install.md), "Prepare the datasets".

### Getting Started
Here is an example of single GPU non-distributed training STC-LIF on Moving MNIST dataset.
> python tools/train.py -d mmnist --lr 1e-3 --min_lr 1e-5 --data_root 'data_dir' -c configs/mmnist/STCLIF.py -e 500 -b 16 --ex_name mmnist_stclif

You can find the experimental configuration file in [configs](configs), and the corresponding model in [convlif_modules.py](openstl/modules/convlif_modules.py).



[1]: https://icml.cc/virtual/2024/poster/33269
[2]: https://github.com/chengtan9907/OpenSTL
[3]: https://github.com/fangwei123456/spikingjelly
