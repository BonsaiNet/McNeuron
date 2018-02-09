 # McNeuron: A Python Package for Sampling Neuron Morphology

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/BonsaiNet/McNeuron/blob/master/LICENSE)[![Gitter](https://badges.gitter.im/glm-tools/pyglmnet.svg)](https://gitter.im/McNeuron/Lobby)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1170488.svg)](https://doi.org/10.5281/zenodo.1170488)

McNeuron is a 'more of this' algorithm for sampling morphologies.  Specifically, it is based on Markov chain Monte Carlo methods for generating morphologies. By iterating changes, the chain becomes closer and closer to the generative distribution until it samples from a meaningful distribution. The advantage of our approach is that it generates morphologies based on a dataset of neurons, compared to other generative models which we can only optimize a small number of generator parameters. In this setup by adding more features or generally by making the generative model better, the resulting morphologies gradually become more realistic.

## Reversible-jump Markov Chain Monte Carlo (RJMCMC):
[RJMCMA](https://en.wikipedia.org/wiki/Reversible-jump_Markov_chain_Monte_Carlo) is the core of sampling in McNeuron. McNeuron first extracts all the desired features from a database of neuron and builds a generative model using na{\"i}ve Bayes assumption. Then by starting from a simple neuron (e.g. a neuron with soma only) and perturbing it using Reversible-jump Markov Chain Monte Carlo in each iteration, gradually enriches the morphology with the desired features.  


![alt tag](https://github.com/BonsaiNet/McNeuron/blob/master/github-overview.png)


## Installation and Usage:
Clone the repository.
```bash
$ git clone https://github.com/BonsaiNet/McNeuron.git
```
To use the algorithm go to 
```
https://github.com/BonsaiNet/McNeuron/blob/master/Generating%20Neurons%20with%20MCMC.ipynb
```
and follow the notebook.
## Dependencies:

- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/)

## Reference:
The preprint (not peer-reviewed) is available at
[Farhoodi, Roozbeh, and Konrad P. Kording. "Sampling Neuron Morphologies." bioRxiv (2018): 248385.](https://www.biorxiv.org/content/early/2018/01/15/248385)

Contact: Roozbeh Farhoodi [roozbehfarhoodi@gmail.com]
