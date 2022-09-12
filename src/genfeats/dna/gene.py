from dataclasses import dataclass

@dataclass(frozen = True)
class Gene:
    
    func: str
    freq: tuple 
    source: tuple
    func_args: tuple