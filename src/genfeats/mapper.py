import numpy as np
from typing import Union
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome

class Mapper:
    
    def __init__(self, features: str) -> None:
        pass
    
    def to_phenotype(self, dna: Union[Gene, Chromesome]) -> np.ndarray:
        if isinstance(dna, Gene):
            phenotype = self.__map_gene(dna)
        elif isinstance(dna, Chromesome):
            phenotype = self.__map_chromesome(dna)
        else:
            raise TypeError("Only Genes and Chromesomes can be maped to a Phenotype")
        return phenotype
    
    def __map_gene(self, gene: Gene) -> np.ndarray:
        pass

    def __map_chromesome(self, chromesome: Chromesome) -> np.ndarray:
        pass