from dataclasses import dataclass
from typing import Any
import json

@dataclass(frozen=True, order=True)
class Gene:
    
    feature: str
    freq_bands: tuple 
    channels: tuple
    feature_parameters: tuple
    
    def __post_init__(self) -> None:
        super().__setattr__('feature', str(self.feature))
        super().__setattr__('freq_bands', tuple(sorted(tuple(band) for band in self.freq_bands)))
        super().__setattr__('channels', tuple(sorted(self.channels)))
        feature_parameters = self.feature_parameters.copy()
        for parameter, value in feature_parameters.items():
            if isinstance(value, list):
                feature_parameters.update({parameter: tuple(value)})
        super().__setattr__('feature_parameters', tuple(sorted(feature_parameters.items())))
        
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
            'feature': self.feature,
            'freq_bands': self.freq_bands,
            'channels': self.channels,
            'feature_parameters': dict(self.feature_parameters)
        }
        return gene

    def to_json(self) -> str:
        return json.dumps(self.to_dict())