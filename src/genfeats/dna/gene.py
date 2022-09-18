from dataclasses import dataclass
from typing import Any
import json

@dataclass(frozen=True)
class Gene:
    
    func: str
    freq: tuple 
    source: tuple
    func_args: tuple
    
    def __post_init__(self) -> None:
        super().__setattr__('func', str(self.func))
        super().__setattr__('freq', tuple(tuple(band) for band in self.freq))
        super().__setattr__('source', tuple(tuple(channel) for channel in self.source))
        func_args = self.func_args.copy()
        for key, value in func_args.items():
            if isinstance(value, list):
                func_args.update({key: tuple(value)})
        super().__setattr__('func_args', tuple(sorted(func_args.items()))) 
    
    @classmethod
    def from_dict(cls, gene: dict):
        return cls(**gene)
    
    @classmethod
    def from_json(cls, gene: str):
        gene = json.loads(gene)
        return cls(**gene)