import json
import numpy as np
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome

class GenotypeBuilder:
    
    def __init__(self, nucleobases: str, chromesome_len: int = 3) -> None:
        with open(nucleobases, "r") as nucleobases_file:
            self.nucleobases = json.load(nucleobases_file)
        self.chr_len = chromesome_len
        self.rng = np.random.default_rng()
        
        self.num_of_channels = len(self.nucleobases['channels'])
        self.num_of_freq_bands = len(self.nucleobases['frequency_bands'])
        self.num_of_features = len(self.nucleobases['features'])
    
    def make_gene(self):
        # Choose channel - frequency combinations option
        toss = self.rng.choice(3)
        if toss == 0:
            channels_config = 2
            freq_bands_config = 1
        elif toss == 1:
            channels_config = 1
            freq_bands_config = 2
        else:
            channels_config = 1
            freq_bands_config = 1
        
        index = self.rng.choice(self.num_of_channels, channels_config, replace=False)
        channels = [self.nucleobases['channels'][i] for i in index]
        
        index = self.rng.choice(self.num_of_freq_bands, freq_bands_config, replace=False)
        freq_bands = [self.nucleobases['frequency_bands'][i] for i in index]
        
        index = self.rng.choice(self.num_of_features)
        feature = list(self.nucleobases['features'].keys())[index]
        
        feature_parameters = dict()
        for parameter, possible_values in self.nucleobases['features'][feature].items():
            index = self.rng.choice(len(possible_values))
            feature_parameters[parameter] = possible_values[index]
            
        return Gene(feature, freq_bands, channels, feature_parameters)
        
    def make_chromesome(self):
        genes = set()
        while len(genes) < self.chr_len:
            genes.add(self.make_gene())
            
        return Chromesome(genes)
    
    def make_population(self, size: int):
        
        return [self.make_chromesome() for _ in range(size)]
    