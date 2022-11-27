import os
import json
import numpy as np
from math import ceil
from math import sqrt
from mne.epochs import BaseEpochs
from numpy.random import default_rng
from scipy.stats import pointbiserialr
from itertools import combinations, chain
from  datetime import datetime as datetime
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
        self.merit_record = dict()
        self.score_records = dict()
        self.progress_counter = 0
        self.mutate_edit_counter = 0
        self.mutation_rates = [0.1,0.2,0.7]
        self.extintion_fate = self.rng.choice(np.arange(30,100,10))
    
        self.epochs = epochs.copy()
        self.filterbank = epochs.copy()
        self.resources_folder = resources_folder
        
        self.folds = 2 if 'folds' not in kwargs.keys() else kwargs['folds']
        self.survival_rate = 0.1 if 'survival_rate' not in kwargs.keys() else kwargs['survival_rate']
        self.chromesome_size = 3 if 'chromesome_size' not in kwargs.keys() else kwargs['chromesome_size']
        self.population_size = 40 if 'population_size' not in kwargs.keys() else kwargs['population_size']
        self.extintions_limit = 10 if 'extintions_limit' not in kwargs.keys() else kwargs['population_size']
        self.generations_limit = 100 if 'generations_limit' not in kwargs.keys() else kwargs['generations_limit']
        self.results_path = './genfeatsBS_results/' if 'results_path' not in kwargs.keys() else kwargs['results_path']
        self.execution_metadata = dict() if 'execution_metadata' not in kwargs.keys() else kwargs['execution_metadata']
        
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
        
        self.results = dict()
        
        self.results['info'] = {
            'survival_rate': self.survival_rate,
            'chromesome_size': self.chromesome_size,
            'population_size': self.population_size,
            'extintions_limit': self.extintions_limit,
            'generations_limit': self.generations_limit,
            'StratifiedKFold_n_splits': self.folds,
            'metadata': self.execution_metadata
        }
        
        self.population = self.genotype_builder.make_population(self.population_size)

        self.extintion = 0
        self.generation = 0
        self.best = self.population[0]
        self.score_records[self.best] = 0
        for i in range(self.generations_limit):
            self.generation = i
            
            self.results[str(self.generation)] = {
                'accuracy': None,
                'solution': None,
                'avg_offspring_merit': None,
                'avg_feature_feature_corr': list()
            }
            
            self.map_population()

            self.rate_population()

            self.select_parents()
        
            self.cross_over()

            self.niche()
        
            self.mutate()

            self.update_mutation_rates()
            
            #print(self.generation, ('%.4f' % ((self.score_records[self.best]+100)/200)), self.best)
            self.results[str(self.generation)]['accuracy'] = float('%.4f' % ((self.score_records[self.best]+100)/200))
            self.results[str(self.generation)]['solution'] = self.best.to_dict()
            self.results[str(self.generation)]['avg_offspring_merit'] = np.mean(list(self.merit_record.values()))
            self.results[str(self.generation)]['avg_feature_feature_corr'] = np.mean(self.results[str(self.generation)]['avg_feature_feature_corr'])
            
            self.set_next_generation()
            
            if self.score_records[self.best] > 80: break
        
        self.save_results()

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
        self.parents = list(self.parents)
    
    def cross_over(self):
        self.merit_record.clear()
        possible_offsprings = combinations(self.genoma, self.chromesome_size)
        offspring = list()
        for genes in possible_offsprings:
            if len(genes) == len(set(genes)):
                offspring.append(Chromesome(genes))
        self.merit_record = {chromesome: self.computeMerit(chromesome) for chromesome in offspring}
        while len(self.merit_record) > self.offspring_size:
            worst = min(self.merit_record, key=(lambda x: self.merit_record[x]))
            del self.merit_record[worst]
        self.offspring = list(self.merit_record.keys())
    
    def mutate(self):
        offspring_genoma = list(chain(*self.offspring))
        offspring = list()
        for chromesome in self.offspring:
            genes = set()
            for gene in chromesome:
                d3 = self.rng.choice(3, p=self.mutation_rates)
                functional = np.count_nonzero(self.cache[gene])
                if not functional or d3 == 0:
                    genes.add(self.genotype_builder.make_gene())
                elif d3 == 1:
                    genes.add(self.genotype_builder.mutate_gene(gene))
                else:
                    genes.add(gene)
            while len(genes) < self.chromesome_size:
                genes.add(self.genotype_builder.make_gene())
            offspring.append(Chromesome(genes))
        self.offspring = offspring   
            
    def niche(self):
        self.niches.clear()
        for chromesome in self.parents:
            genes = set()
            for gene in chromesome:
                initial_size = len(genes)
                while len(genes) == initial_size: 
                    genes.add(self.genotype_builder.mutate_gene(gene))
            self.niches.append(Chromesome(genes))
        
    def computeMerit(self, candidate):
        X = self.make_X(candidate).transpose()
        n = self.chromesome_size
        y = self.epochs.events[:, -1]
        avg_feature_class_corr = np.average([pointbiserialr(x, y).correlation for x in X])
        avg_feature_feature_corr = ((np.abs(np.corrcoef(X)).sum()-n)/2)/((n**2 - n)/2)
        merit = (n*avg_feature_class_corr)/sqrt(n+n*(n-1)*avg_feature_feature_corr)
        self.results[str(self.generation)]['avg_feature_feature_corr'].append(avg_feature_feature_corr)
        return merit
        
    def make_X(self, chromesome) -> np.ndarray:
        x = np.empty((len(self.epochs), self.chromesome_size))
        for g in range(self.chromesome_size):
            gene = chromesome[g]
            x[:, g] = self.cache.get(gene)
        return x

    def update_mutation_rates(self):

        new_best = max(self.score_records, key=(lambda x: self.score_records[x]))
        progress = self.score_records[new_best] - self.score_records[self.best]
                
        if progress < 0.1:
            self.progress_counter += 1
            if self.mutation_rates[1]<0.35:
                self.mutation_rates[1] += 0.050
                self.mutation_rates[2] -= 0.050
            elif self.mutation_rates[2] > 0.3:
                self.mutation_rates[0] += 0.050
                self.mutation_rates[2] -= 0.050
            else:
                if self.mutate_edit_counter>=3:
                    self.mutation_rates = [0.1,0.2,0.7] 
                    self.mutate_edit_counter = 0
                else: 
                    self.mutate_edit_counter +=1 
        elif progress > 0.1:
            self.progress_counter -= 1 if progress < 1 else int(progress)
            if self.progress_counter < 0: self.progress_counter = 0

            if self.mutation_rates[2]<0.5:
                self.mutation_rates[0] -= 0.050
                self.mutation_rates[2] += 0.050
            elif self.mutation_rates[2]<0.70:
                self.mutation_rates[1] -= 0.050
                self.mutation_rates[2] += 0.050
            else:
                if self.mutate_edit_counter>=3:
                    self.mutation_rates = [0.1,0.2,0.7] 
                    self.mutate_edit_counter = 0
                else: 
                    self.mutate_edit_counter +=1
        self.best = new_best

    def set_next_generation(self):
        if self.progress_counter <= self.extintion_fate:
            self.population = self.offspring + self.parents + self.niches
        elif self.extintion < self.extintions_limit:
            self.population = self.genotype_builder.make_population(self.population_size - 1 )
            self.extintion_fate = self.rng.choice(np.arange(50,100,10))
            self.population.append(self.best)
            self.progress_counter = 0
            self.extintion += 1
        else:
            self.population = self.offspring + self.parents + self.niche
    
    def save_results(self):
        
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        
        if 'subject' in self.execution_metadata:
            subject = self.execution_metadata['subject']
            results_file_name = f'results-{subject}-{timestamp}.json'
        else:
            results_file_name = self.results_path + f'results-{timestamp}.json'
  
        with open(results_file_name, 'w') as write_file:
            json.dump(self.results, write_file, indent=4)