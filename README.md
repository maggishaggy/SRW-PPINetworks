# SRW-PPINetworks


This repository is an implementation of the [Supervised Random Walks algorithm](https://cs.stanford.edu/people/jure/pubs/linkpred-wsdm11.pdf) (MATLAB original code [link](https://github.com/syllogismos/facebook-link-prediction)) for Protein-Protein Interaction Networks. The algorithm is implemented in Python where the optimization is implemented using the function *fmin_l_bfgs_b* from Scipy module with Wilcoxon-Mann-Whitney (WMW) loss function. Learning the parameter vector **w** can be done with the function *supervised_random_walks*, and with the parameter vector **w**, the function *random_walks* gives the random walk parameter vector **p**. The alternative implementation of the algorithm is implemented to work on GPU using the NDArray API of [MXNet](https://mxnet.incubator.apache.org/).

### Scipts:
- *supervised_random_walks.py* - implementation of the SRW on CPU
- *supervised_random_walks_gpu.py* - implementation of the SRW on GPU
- *train.py* - train and test the SRW algorithm
- Preprocessing scripts:
  - *find_largest_component.py*
  - *min_max_normalization.py*
  - *semanticSimilarityGoTerms.R* and *semanticSimilarityGoTerms.py* - calculation of semantic similarity of GO terms
  - *semanticSimilarityProteins.R* and *semanticSimilarityProteins.py* - calculation of semantic similarity of proteins
