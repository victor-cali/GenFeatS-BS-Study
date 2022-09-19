from src.genfeats.dna.gene import Gene 
from dataclasses import dataclass
from typing import Tuple
import json

@dataclass(frozen = True)
class Chromesome:
    genes: Tuple[Gene]
    
    def __post_init__(self) -> None:
        assert len(self.genes) == len(set(self.genes)) 
        super().__setattr__('genes', tuple(sorted(self.genes)))
    
    def __str__(self) -> str:
        return self.to_json()
    
    def __iter__(self):
        super().__setattr__('_g', 0)
        return self

    def __len__(self) -> int:
        return len(self.genes)
    
    def __next__(self):
        if self._g < self.__len__():
            super().__setattr__('_g', self._g + 1)
            return self.genes[self._g-1]
        else:
            raise StopIteration
    
    def __getitem__(self, i):
        return self.genes[i]

    @classmethod
    def from_dict(cls, chromesome: dict):
        return cls(tuple(Gene.from_dict(gene) for gene in chromesome.values()))
    
    @classmethod
    def from_json(cls, chromesome: str):
        chromesome = json.loads(chromesome)
        return cls.from_dict(chromesome)
    
    def to_dict(self) -> list[dict]:
        length = range(1, len(self.genes) + 1)
        return {f'g{i}': g.to_dict() for g, i in zip(self.genes, length)}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())