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


We encourage you to report issues at our `Github page`_ ; you can also create pull reports there to contribute your enhancements.
If Velorama is useful for your research, please consider citing `bioRxiv (2022)`_.

.. _bioRxiv (2022): https://www.biorxiv.org/content/10.1101/TBD
.. _Github page: https://github.com/rs239/velorama
