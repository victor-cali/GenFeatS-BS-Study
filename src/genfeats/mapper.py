import ray
import numpy as np
from mne.epochs import BaseEpochs
from typing import Union
from typing import Callable
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome
from inspect import getfullargspec, getmembers, isfunction
from importlib.machinery import SourceFileLoader

class Mapper:
    
    def __init__(self, features: str, epochs: BaseEpochs) -> None:
        self.features = SourceFileLoader('features', features).load_module()
        self.features = dict(getmembers(self.features, isfunction))
        self.epochs = epochs
        self.sfreq = epochs.info['sfreq']
    
    def to_phenotype(self, dna: Union[Gene, Chromesome]) -> np.ndarray:
        if isinstance(dna, Gene):
            gene = self.__prepare_mapping(dna)
            reference = map.remote(f=self.features[gene['feature']], kwargs=gene['feature_parameters'])
            phenotype = ray.get(reference)
        elif isinstance(dna, Chromesome):
            chromesome_size = len(dna)
            phenotype = np.empty((len(epochs), chromesome_size))
            references = []
            for i in range(chromesome_size):
                gene = self.__prepare_mapping(dna[i])
                reference = map.remote(f=self.features[gene['feature']], kwargs=gene['feature_parameters'])
                references.append(reference)
            for i in range(chromesome_size):
                phenotype[:, i] = ray.get(references[i])
        else:
            raise TypeError("Only Genes and Chromesomes can be maped to a Phenotype")
        return phenotype
    
    def __prepare_mapping(self, gene: Gene) -> dict:
        gene = gene.to_dict()
        if 'freq_band' in set(getfullargspec(features['f1'])[0]):
            gene['feature_parameters']['freq_band'] = gene['freq_bands']
            data = gene['channels']
        else:
            channels = gene['channels']
            freq_bands = gene['freq_bands']
            data = [f'{ch}({fb[0]}-{fb[1]})' for ch in channels for fb in freq_bands]
        if 'sfreq' in set(getfullargspec(features['f1'])[0]):
            gene['feature_parameters']['sfreq'] = self.sfreq
        gene['feature_parameters']['data'] = self.epochs.get_data(picks=data)
        return gene

@ray.remote
def map(f: Callable, kwargs: dict):
    return f(**kwargs)