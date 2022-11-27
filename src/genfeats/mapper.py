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
        
    def to_phenotype(self, dna: Union[Gene, Chromesome], return_dict: bool = False) -> np.ndarray:
        if isinstance(dna, Gene):
            phenotype = self.__gene_to_phenotype(dna, return_dict=return_dict)
        elif isinstance(dna, Chromesome):
            phenotype = self.__chromesome_to_phenotype(dna, return_dict=return_dict)
        elif isinstance(dna, list) or isinstance(dna, tuple):
            phenotype = self.__population_to_phenotype(dna)
        else:
            raise TypeError("Only Genes and Chromesomes can be maped to a Phenotype")
        return phenotype

    def __gene_to_phenotype(self, gene: Gene, return_dict: bool) -> np.ndarray:
        gene_dto = self.__prepare_mapping(gene)
        reference = self.pool.starmap(
            self.features_handler.map.remote,
            [(gene_dto['data'], gene_dto['feature'], gene_dto['feature_parameters'])]
        )
        if return_dict:
            phenotype = dict(gene = ray.get(reference[0]))
        else:
            phenotype = ray.get(reference[0])
        return phenotype
    
    def __chromesome_to_phenotype(self, chromesome: Chromesome, return_dict: bool) -> np.ndarray:
        args_list = []
        n_genes = len(chromesome)
        phenotype = dict() if return_dict else np.empty((len(self.epochs), n_genes))
        for i in range(n_genes):
            gene_dto = self.__prepare_mapping(chromesome[i])
            args_list.append((gene_dto['data'], gene_dto['feature'], gene_dto['feature_parameters']))
        references = self.pool.starmap(
            self.features_handler.map.remote,
            args_list
        )
        for i in range(n_genes):
            if return_dict:
                phenotype[chromesome[i]] = ray.get(references[i])
            else:
                phenotype[:, i] = ray.get(references[i])
        return phenotype
    
    def __population_to_phenotype(self, population: Union[list, tuple]) -> np.ndarray:
        args_list = []
        n_genes = len(population)
        for i in range(n_genes):
            gene_dto = self.__prepare_mapping(population[i])
            args_list.append((gene_dto['data'], gene_dto['feature'], gene_dto['feature_parameters']))
        references = self.pool.starmap(
            self.features_handler.map.remote,
            args_list
        )
        for i in range(n_genes):
            phenotype = {population[i]: ray.get(references[i])}
            yield phenotype
    
    def __prepare_mapping(self, gene: Gene) -> dict:
        gene = gene.to_dict()
        if 'frequency_bands' in self.features_args[gene['feature']]:
            gene['feature_parameters']['frequency_bands'] = gene['frequency_bands']
            data = gene['channels']
        else:
            channels = gene['channels']
            frequency_bands = gene['frequency_bands']
            data = [f'{ch}({fb[0]}-{fb[1]})' for ch in channels for fb in frequency_bands]
        if 'sfreq' in self.features_args[gene['feature']]:
            gene['feature_parameters']['sfreq'] = self.sfreq
        gene['data'] = self.epochs.get_data(picks=data)
        del gene['channels']
        del gene['frequency_bands']
        return gene

@ray.remote
class FeaturesHandler():
    def __init__(self, features_file: str) -> None:
        module_name = 'nucleobases'
        spec = importlib.util.spec_from_file_location('nucleobases', features_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules['nucleobases'] = module
        spec.loader.exec_module(module)
        self.features = dict(getmembers(module, isfunction))
        self.features_args = {key: getfullargspec(value)[0] for key, value in self.features.items()}
        
    def map(self, data, feature, kwargs):
        n_epochs = data.shape[0]
        phenotype = np.empty(n_epochs)
        for i in range(n_epochs):
            phenotype[i] = self.features[feature](data=data[i,:], **kwargs)
            
        if np.any(np.isnan(phenotype)) or np.any(np.isinf(phenotype)):
            try:
                np.nan_to_num(phenotype, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    val = np.nanmean(phenotype)
                np.nan_to_num(phenotype, copy=False, nan=val, posinf=val, neginf=val)
            except Exception as e:
                np.nan_to_num(phenotype, copy=False, nan=0, posinf=0, neginf=0)
            
        return phenotype

    def get_features_args(self):
        return self.features_args
    
