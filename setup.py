from setuptools import setup

setup(name='velorama',
      version='0.0.1',
      description='Gene regulatory network inference for RNA velocity and pseudotime data',
      url='http://github.com/rs239/velorama',
      author='Anish Mudide, Alex Wu, Rohit Singh',
      author_email='rsingh@alum.mit.edu',
      license='MIT',
      packages=['velorama'],
      install_requires = 'numpy,scipy,pandas,sklearn,cellrank,scvelo,ray'.split(','),
      scripts = ['bin/velorama'],
      zip_safe=False)
