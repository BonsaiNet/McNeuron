 # McNeuron
 [![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/BonsaiNet/McNeuron/blob/master/LICENSE)[![Gitter](https://badges.gitter.im/glm-tools/pyglmnet.svg)](https://gitter.im/McNeuron/Lobby)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1170488.svg)](https://doi.org/10.5281/zenodo.1170488)
McNeuron is a Python Package to Utilize [Neuron Morphology](https://en.wikipedia.org/wiki/Neuromorphology). It can visualize neurons, extract their feature, and generate novel samples. 




## Installation:
Clone the repository.
```bash
$ git clone https://github.com/BonsaiNet/McNeuron.git
```

## Usage
### downloading neuromorpho.org database
For more information Go to the following notebook with title: `Download neuromorpho.org`

### Reading and plotting swc files
For more information Go to the following notebook with title: `How to use McNeuron to depict a neuron`

### Extracting features of a neuron
For more information Go to the following notebook with title: `Extracting features`

###  Sampling neuron morphology:
We can generate samples of different cell types by McNeuron. Technically, it uses Reversible-jump Markov Chain Monte Carlo ([RJMCMA](https://en.wikipedia.org/wiki/Reversible-jump_Markov_chain_Monte_Carlo)) generating morphologies. It extracts features of a database of neuron and then generates neuron with the same features. Specificaly, the algorithm starts from a simple neuron (e.g. a neuron with soma only) and perturbing it using RJMCMC and gradually enriches the morphology with the desired features (see figure below). In this framework, adding more morpgological features reasults in more realistic neurons. The main advantage of this approach is that it can generates morphologies for any cell type.

![alt tag](https://github.com/BonsaiNet/McNeuron/blob/master/github-overview.png)

You can see a few samples here:

If you want to know more about this algorithm read the paper below:
[Farhoodi, Roozbeh, and Konrad P. Kording. "Sampling Neuron Morphologies." bioRxiv (2018): 248385.](https://www.biorxiv.org/content/early/2018/01/15/248385)

To run the algorithm follow the script go to 
```
https://github.com/BonsaiNet/McNeuron/blob/master/Generating%20Neurons%20with%20MCMC.ipynb
```

## Dependencies:

- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/)
- [scikit learn](https://scikit-learn.org/stable/)

## References:
If you used this package, we really appriciate if you cite our paper:
[Farhoodi, Roozbeh, and Konrad P. Kording. "Sampling Neuron Morphologies." bioRxiv (2018): 248385.](https://www.biorxiv.org/content/early/2018/01/15/248385)

Contact: Roozbeh Farhoodi [roozbehfarhoodi@gmail.com]
