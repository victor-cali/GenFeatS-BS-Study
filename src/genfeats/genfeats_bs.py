import numpy as np
from math import ceil
from math import sqrt
from mne.epochs import BaseEpochs
from numpy.random import default_rng
from scipy.stats import pointbiserialr
from itertools import combinations, chain
from sklearn.model_selection import StratifiedKFold

from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome
from src.genfeats.genotype_builder import GenotypeBuilder
from src.genfeats.mapper import Mapper

class GenFeatSBS:
    
    def __init__(self, resources_folder: str, epochs: BaseEpochs, fitness_function, **kwargs) -> None:
        self.cache = dict()
        self.niches = list()
        self.offspring = list()
        self.genoma: np.ndarray
        self.rng = default_rng()
        self.score_records = dict()
    
        self.epochs = epochs.copy()
        self.filterbank = epochs.copy()
        self.resources_folder = resources_folder
        
        self.folds = 2 if 'folds' not in kwargs.keys() else kwargs['folds']
        self.survival_rate = 0.1 if 'survival_rate' not in kwargs.keys() else kwargs['survival_rate']
        self.chromesome_size = 4 if 'chromesome_size' not in kwargs.keys() else kwargs['chromesome_size']
        self.population_size = 100 if 'population_size' not in kwargs.keys() else kwargs['population_size']
        self.extintions_limit = 10 if 'extintions_limit' not in kwargs.keys() else kwargs['population_size']
        self.generations_limit = 100 if 'generations_limit' not in kwargs.keys() else kwargs['generations_limit']
        self.results_path = './genfeatsBS_results/' if 'results_path' not in kwargs.keys() else kwargs['results_path']
        
        nucleobases = resources_folder + 'nucleobases.json'
        self.genotype_builder = GenotypeBuilder(nucleobases=nucleobases, chromesome_size=self.chromesome_size)
        
        self.__set_filterbank()
        
        features_file = resources_folder + 'nucleobases.py'
        self.map = Mapper(features_file=features_file, chromesome_size=self.chromesome_size, epochs=self.filterbank)
        
        self.skf = StratifiedKFold(n_splits=self.folds)
        
        self.fitness_function = fitness_function
        
        self.num_of_parents = int(self.population_size * self.survival_rate)
        self.tournament_size = ceil(self.num_of_parents/2)
        self.offspring_size = int(self.population_size - self.num_of_parents * 2)
        
    def __call__(self, *args: any, **kwds: any) -> any:
        
        self.population = self.genotype_builder.make_population(self.population_size)
        
        self.generation = 0
        while self.generation < self.generations_limit:
            
            self.map_population()

            self.rate_population()

            self.select_parents()
        
            self.cross_over()
            
            return self.offspring

            self.niche()
        
            self.mutate()

            self.update_mutation_rates()

            self.set_next_generation()

            self.record_generation()

            print(f'Generation: {self.generation}, Extintions: {self.extintion}, Best: {self.best.score}')
        print(f'FINISHED\n Best Candidate: {self.best.score}')
        return self.best
    
    def __set_filterbank(self):
        new_names = dict()
        filtered_epochs = list()
        for fb in self.genotype_builder.nucleobases['frequency_bands']:
            new_names.clear()
            for ch in self.genotype_builder.nucleobases['channels']:
                new_names.update({ch: f'{ch}({fb[0]}-{fb[1]})'})
            subepochs = self.epochs.copy()
            subepochs.filter(fb[0], fb[1], method = 'iir', verbose = 50)
            subepochs.rename_channels(new_names)
            filtered_epochs.append(subepochs.copy())
        self.filterbank.add_channels(filtered_epochs,force_update_info=True)
    
    def map_population(self) -> None:
        stream_population = set([gene for chromesome in self.population for gene in chromesome])
        self.cache = {key: self.cache[key] for key in set(self.cache) & stream_population}
        stream_population = list(stream_population - set(self.cache))
        for entry in self.map.to_phenotype(stream_population):
            self.cache.update(entry)
    
    def rate_population(self) -> None:
        self.score_records = {key: self.score_records[key] for key in set(self.score_records) & set(self.population)}
        for chromesome in self.population:
            if chromesome not in self.score_records:
                fitness = 0
                X = self.make_X(chromesome)
                y = self.epochs.events[:, -1]
                for train_index, test_index in self.skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    #Train the model using the training sets
                    try:
                        self.fitness_function.fit(X_train, y_train)
                        accuracy = self.fitness_function.score(X_test, y_test)
                    except:
                        accuracy = 0.5
                    fitness += accuracy
                fitness /= self.folds
                score = 200.0*fitness-100.0
                self.score_records.update({chromesome: score})

    def select_parents(self) -> None:
        self.parents = set()
        guide = self.score_records.copy()
        best = max(guide, key=(lambda x: guide[x]))
        self.parents.add(best)
        del guide[best]
        for _ in range(self.num_of_parents-1):
            candidates = np.array(list(guide.items()), dtype=object)
            selected = dict(self.rng.choice(candidates, self.tournament_size, replace=False))
            best = max(selected, key=(lambda x: selected[x]))
            self.parents.add(best)
            del guide[best]
        self.genoma = np.array(list(chain(*self.parents)), dtype=object)
    
    def cross_over(self):
        offspring = [Chromesome(genes) for genes in combinations(self.genoma, self.chromesome_size)]
        merit_record = {chromesome: self.computeMerit(chromesome) for chromesome in offspring}
        while len(merit_record) > self.offspring_size:
            worst = min(merit_record, key=(lambda x: merit_record[x]))
            del merit_record[worst]
        self.offspring = list(merit_record.keys())
    
    def mutate(self):
        offspring_genoma = list(chain(*self.offspring))
        for i in range(len(offspring_genoma)):
            d3 = self.rng.choice(3)
            if d3 == 0:
                gene = offspring_genoma[i]
                offspring_genoma[i] = self.genotype_builder.mutate_gene(gene)
            elif d3 == 1:
                gene = offspring_genoma[i]
                offspring_genoma[i] = self.genotype_builder.make_gene()
        
    def niche(self):
        self.niches.clear()
        for chromesome in self.parents:
            genes = [self.genotype_builder.mutate_gene(gene) for gene in chromesome]
            self.niches.append(Chromesome(genes))
        
    def computeMerit(self, candidate):
        X = self.make_X(candidate).transpose()
        n = self.chromesome_size
        y = self.epochs.events[:, -1]
        avg_feature_class_corr = np.average([pointbiserialr(x, y).correlation for x in X])
        avg_feature_feature_corr = ((np.abs(np.corrcoef(X)).sum()-n)/2)/((n**2 - n)/2)
        merit = (n*avg_feature_class_corr)/sqrt(n+n*(n-1)*avg_feature_feature_corr)
        return merit
        
    def make_X(self, chromesome) -> np.ndarray:
        x = np.empty((len(self.epochs), self.chromesome_size))
        for g in range(self.chromesome_size):
            gene = chromesome[g]
            x[:, g] = self.cache.get(gene)
        return x
        