
from .models import VeloramaMLP
from .train import train_model
from .run import execute_cmdline

__all__ = ['VeloramaMLP', 'train_model', 'execute_cmdline']


import pkgutil, pathlib, importlib

# from pkgutil import iter_modules
# from pathlib import Path
# from importlib import import_module

# https://julienharbulot.com/python-dynamical-import.html
# iterate through the modules in the current package
#
# # package_dir = pathlib.Path(__file__).resolve().parent
# # for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
# #     if 'datasets' in module_name:
# #         module = importlib.import_module(f"{__name__}.{module_name}")
