|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/velorama_v5.png
   :target: https://pypi.org/project/velorama



Velorama - Gene regulatory network inference for RNA velocity and pseudotime data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Velorama is a Python library for inferring gene regulatory networks from single-cell RNA-seq data

**It is designed for the case where RNA velocity or pseudotime data is available.**
Here are some of the analyses that you can do with Velorama:

  - infer temporally-causal regulator-target links from RNA velocity cell-to-cell transition matrices. 
  - infer over branching/merging trajectories, with just pseudotime data, without having to manually separate them.
  - estimate the relative speed of various regulators (i.e., how quickly they act on the target.
    
Velorama offers support for both pseudotime and RNA velocity data. 


Velorama is based on a Granger causal approach and models the differentiation landscape as a directed acyclic graph of cells, rather than as a linear ordering that previous approaches have done.

## API Example Usage
Velorama is currently implemented as a command line tool that operates on AnnData objects. First, prepare an AnnData object of the dataset to be analyzed with Velorama and annotate genes that are to be analyzed as candidate regulators and candidate targets as follows. Here, ```regulator_genes``` is the set of gene symbols or IDs for the candidate regulators, while ```target_genes``` indicates the set of gene symbols or IDs for the candidate target genes. This AnnData object should be saved as ```{dataset}.h5ad```.

```python
adata.var['is_reg'] = [n in regulator_genes for n in adata.var.index.values]
adata.var['is_target'] = [n in target_genes for n in adata.var.index.values]
```

The below command runs Velorama, which saves the outputs for inferring Granger causal interactions and interaction speeds to a selected directory.
```bash
velorama -m $method -ds $dataset -dyn $dynamics -dev $device -l $L -hd $hidden -rd $rd 
```
Here, ```$dataset``` is the file name of the saved AnnData object. ```$dynamics``` can be "rna_velocity" or "pseudotime", depending on which data the user desires to use to construct the DAG. ```$device``` is chosen to be either "cuda" or "cpu". ```$L``` refers to the maximum number of lags to consider. ```$hidden``` indicates the dimensionality of the hidden layers. ```$rd``` indicates the name of the root directory that contains the saved AnnData object and where the outputs will be saved. 

We encourage you to report issues at our `Github page`_ ; you can also create pull reports there to contribute your enhancements.
If Velorama is useful for your research, please consider citing `bioRxiv (2022)`_.

.. _bioRxiv (2022): https://www.biorxiv.org/content/10.1101/TBD
.. _Github page: https://github.com/rs239/velorama
