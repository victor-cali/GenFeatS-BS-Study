import ray
from ray.util.multiprocessing import Pool
import numpy as np
from mne.epochs import BaseEpochs
from typing import Union
from typing import Callable
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome
from inspect import getfullargspec, getmembers, isfunction
import importlib.util
import sys

class Mapper:
    
    def __init__(self, features_file: str, chromesome_size: int, epochs: BaseEpochs) -> None:
        self.pool = Pool()
        self.epochs = epochs
        self.sfreq = epochs.info['sfreq']
        self.features_handler = FeaturesHandler.remote(features_file)
        self.features_args = ray.get(self.features_handler.get_features_args.remote())
        
    def to_phenotype(self, dna: Union[Gene, Chromesome]) -> np.ndarray:
        if isinstance(dna, Gene):
            phenotype = self.__gene_to_phenotype(dna)
        elif isinstance(dna, Chromesome):
            phenotype = self.__chromesome_to_phenotype(dna)
        else:
            raise TypeError("Only Genes and Chromesomes can be maped to a Phenotype")
        return phenotype

    def __gene_to_phenotype(self, gene: Gene) -> np.ndarray:
        gene = self.__prepare_mapping(gene)
        reference = self.pool.starmap(
            self.features_handler.map.remote,
            [(gene['data'], gene['feature'], gene['feature_parameters'])]
        )
        phenotype = ray.get(reference[0])
        return phenotype
    
    def __chromesome_to_phenotype(self, chromesome: Chromesome) -> np.ndarray:
        mylist = []
        n_genes = len(chromesome)
        phenotype = np.empty((len(self.epochs), n_genes))
        for i in range(n_genes):
            gene = self.__prepare_mapping(chromesome[i])
            mylist.append((gene['data'], gene['feature'], gene['feature_parameters']))
        references = self.pool.starmap(
            self.features_handler.map.remote,
            mylist
        )
        for i in range(n_genes):
            phenotype[:, i] = ray.get(references[i])
        return phenotype
    
    def __prepare_mapping(self, gene: Gene) -> dict:
        gene = gene.to_dict()
        if 'freq_band' in self.features_args[gene['feature']]:
            gene['feature_parameters']['freq_band'] = gene['freq_bands']
            data = gene['channels']
        else:
            channels = gene['channels']
            freq_bands = gene['freq_bands']
            data = [f'{ch}({fb[0]}-{fb[1]})' for ch in channels for fb in freq_bands]
        if 'sfreq' in self.features_args[gene['feature']]:
            gene['feature_parameters']['sfreq'] = self.sfreq
        gene['data'] = self.epochs.get_data(picks=data)
        del gene['channels']
        del gene['freq_bands']
        return gene

@ray.remote
class FeaturesHandler():
    def __init__(self, features_file: str) -> None:
        module_name = 'gfsbs_features'
        spec = importlib.util.spec_from_file_location('gfsbs_features', features_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules['gfsbs_features'] = module
        spec.loader.exec_module(module)
        self.features = dict(getmembers(module, isfunction))
        self.features_args = {key: getfullargspec(value)[0] for key, value in self.features.items()}
        
    def map(self, data, feature, kwargs):
        n_epochs = data.shape[0]
        phenotype = np.empty(n_epochs)
        for i in range(n_epochs):
            phenotype[i] = self.features[feature](data=data[i,:], **kwargs)
        return phenotype

    def get_features_args(self):
        return self.features_args
    
