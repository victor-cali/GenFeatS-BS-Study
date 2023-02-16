import json
import numpy as np
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome

class GenotypeBuilder:
    
    def __init__(self, nucleobases: str, chromesome_size: int = 3) -> None:
        with open(nucleobases, "r") as nucleobases_file:
            self.nucleobases = json.load(nucleobases_file)
            
        self.chromesome_size = chromesome_size
        self.rng = np.random.default_rng()
        
        self.num_of_channels = len(self.nucleobases['channels'])
        self.num_of_frequency_bands = len(self.nucleobases['frequency_bands'])
        self.num_of_features = len(self.nucleobases['features'])
    
    def make_gene(self):
        index = self.rng.choice(self.num_of_features)
        feature = list(self.nucleobases['features'].keys())[index]
        
        # Choose channel - frequency combinations option
        if 'constrains' in self.nucleobases['features'][feature]:
            channels_config = self.nucleobases['features'][feature]['constrains']['channels']
            frequency_bands_config = self.nucleobases['features'][feature]['constrains']['frequency_bands']
        else:
            toss = self.rng.choice(3)
            if toss == 0:
                channels_config = 2
                frequency_bands_config = 1
            elif toss == 1:
                channels_config = 1
                frequency_bands_config = 2
            else:
                channels_config = 1
                frequency_bands_config = 1
        
        index = self.rng.choice(self.num_of_channels, channels_config, replace=False)
        channels = [self.nucleobases['channels'][i] for i in index]
        
        index = self.rng.choice(self.num_of_frequency_bands, frequency_bands_config, replace=False)
        frequency_bands = [self.nucleobases['frequency_bands'][i] for i in index]
        if len(frequency_bands) == 2:
            while max(0, min(frequency_bands[0][1], frequency_bands[1][1]) - max(frequency_bands[0][0], frequency_bands[1][0])) != 0:
                index = self.rng.choice(self.num_of_frequency_bands, frequency_bands_config, replace=False)
                frequency_bands = [self.nucleobases['frequency_bands'][i] for i in index]
        
        feature_parameters = dict()
        for parameter, possible_values in self.nucleobases['features'][feature].items():
            if parameter == 'constrains':
                pass
            else:
                index = self.rng.choice(len(possible_values))
                feature_parameters[parameter] = possible_values[index]
                
        args = {
            'feature': feature, 
            'channels': channels, 
            'frequency_bands': frequency_bands,
            'feature_parameters': feature_parameters
        }

        return Gene(**args)
        
    def make_chromesome(self):
        genes = set()
        while len(genes) < self.chromesome_size:
            genes.add(self.make_gene())
        return Chromesome(genes)
    
    def make_population(self, size: int, return_stream: bool = False):
        if return_stream:
            population = [self.make_gene() for _ in range(size * self.chromesome_size)]
        else:
            population = [self.make_chromesome() for _ in range(size)]
        return population

    def mutate_gene(self, gene: Gene):

        gene = gene.to_dict()
        
        feature = gene['feature']
        
        # Choose channel - frequency combinations option
        if 'constrains' in self.nucleobases['features'][feature]:
            channels_config = self.nucleobases['features'][feature]['constrains']['channels']
            frequency_bands_config = self.nucleobases['features'][feature]['constrains']['frequency_bands']
        else:
            toss = self.rng.choice(3)
            if toss == 0:
                channels_config = 2
                frequency_bands_config = 1
            elif toss == 1:
                channels_config = 1
                frequency_bands_config = 2
            else:
                channels_config = 1
                frequency_bands_config = 1

        index = self.rng.choice(self.num_of_channels, channels_config, replace=False)
        channels = [self.nucleobases['channels'][i] for i in index]
        
        index = self.rng.choice(self.num_of_frequency_bands, frequency_bands_config, replace=False)
        frequency_bands = [self.nucleobases['frequency_bands'][i] for i in index]
        if len(frequency_bands) == 2:
            while max(0, min(frequency_bands[0][1], frequency_bands[1][1]) - max(frequency_bands[0][0], frequency_bands[1][0])) != 0:
                index = self.rng.choice(self.num_of_frequency_bands, frequency_bands_config, replace=False)
                frequency_bands = [self.nucleobases['frequency_bands'][i] for i in index]

        feature_parameters = gene['feature_parameters']
 
        args = {
            'feature': feature, 
            'channels': channels, 
            'frequency_bands': frequency_bands,
            'feature_parameters': feature_parameters
        }

        return Gene(**args)