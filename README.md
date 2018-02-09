# McNeuron
### A Python Pachage for Sampling Neuron Morphology

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/BonsaiNet/McNeuron/blob/master/LICENSE)[![Gitter](https://badges.gitter.im/glm-tools/pyglmnet.svg)](https://gitter.im/McNeuron/Lobby)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1170488.svg)](https://doi.org/10.5281/zenodo.1170488)

McNeuron is a 'more of this' method for sampling morphologies. It extracts all the desired features from a database of neuron and build a generative model using na{\"i}ve Bayes assumption. Then by starting from a simple neuron (e.g. a neuron with soma only) and perturb it by Reversible-jump Markov Chain Monte Carlo in each iteration, forces it gradually to capture the desired features. We showed that by running this method on various database of morphologies we are able to automatically sample from them.  

Contact: Roozbeh Farhoodi [roozbehfarhoodi@gmail.com]

![alt tag](https://github.com/BonsaiNet/McNeuron/blob/master/github-overview.png)

Reference:
--------------------
[Sampling Neuron Morphology](https://www.biorxiv.org/content/early/2018/01/15/248385)
Farhoodi, R, and Kording K. P (2018). bioRxiv

Installation:
--------------------
