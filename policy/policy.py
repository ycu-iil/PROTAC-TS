from abc import ABC, abstractmethod

# ----------------------------------------------------------------------
# This file is derived from ChemTSv2
#   https://github.com/molecule-generator-collection/ChemTSv2
#
# MIT License
# Ref. ChemTSv2: Functional molecular design using de novo molecule generator
# WIREs Computational Molecular Science 13, e1680, (2023). https://doi.org/10.1002/wcms.1680
# ----------------------------------------------------------------------

class Policy(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(child_state, conf):
        raise NotImplementedError('Please check your reward file')
    
