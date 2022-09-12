from dataclasses import dataclass

from src.genfeats.dna.gene import Gene 

@dataclass(frozen = True)
class Chromesome:
    genes: tuple[Gene]