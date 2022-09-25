from dataclasses import dataclass
from typing import Any
import json

@dataclass(frozen=True, order=True)
class Gene:
    
    func: str
    freq: tuple 
    source: tuple
    func_args: tuple
    
    def __post_init__(self) -> None:
        super().__setattr__('func', str(self.func))
        super().__setattr__('freq', tuple(sorted(tuple(band) for band in self.freq)))
        super().__setattr__('source', tuple(sorted(self.source)))
        func_args = self.func_args.copy()
        for key, value in func_args.items():
            if isinstance(value, list):
                func_args.update({key: tuple(value)})
        super().__setattr__('func_args', tuple(sorted(func_args.items())))
        
    def  __str__(self) -> str:
        return self.to_json()
    
    @classmethod
    def from_dict(cls, gene: dict):
        return cls(**gene)
    
    @classmethod
    def from_json(cls, gene: str):
        gene = json.loads(gene)
        return cls.from_dict(gene)

    def to_dict(self) -> dict:
        gene = {
            'func': self.func,
            'freq': self.freq,
            'source': self.source,
            'func_args': dict(self.func_args)
        }
        return gene

    def to_json(self) -> str:
        return json.dumps(self.to_dict())