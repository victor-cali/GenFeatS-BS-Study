import json
import pytest
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome

class TestGene:

    @pytest.fixture
    def dict_gene(self) -> dict:
        gene = {
            'feature': 'dummy_func',
            'frequency_bands': [[4, 7.5], [12, 20]],
            'channels': ['C3', 'C4'],
            'feature_parameters': {
                'dummy_arg1': 1,
                'dummy_arg2': 2
            } 
        }
        return gene
    
    @pytest.fixture
    def json_gene(self, dict_gene) -> str:
        return json.dumps(dict_gene)
    
    def test_gene(self, dict_gene):
        gene = Gene(**dict_gene)
        hash(gene)
        
    def test_gene_from_dict(self, dict_gene):
        gene = Gene.from_dict(dict_gene)
        hash(gene)
        
    def test_gene_from_json(self, json_gene):
        gene = Gene.from_json(json_gene)
        hash(gene)
    
    def test_gene_to_json(self, dict_gene):
        gene = Gene(**dict_gene)
        assert isinstance(gene.to_json(), str)

class TestChromesome:
    
    @pytest.fixture
    def genes(self) -> tuple[Gene]:
        g0 = Gene('f1', ('B', 'A'), ((12, 20), (4, 7)), {'a': 1, 'b': 2})
        g1 = Gene('f1', ('A', 'B'), ((4, 7), (12, 20)), {'a': 1, 'b': 2})
        g2 = Gene('f2', ('A', 'C'), ((4, 7), (15, 20)), {'a': 3, 'c': 4})
        g3 = Gene('f2', ('B', 'C'), ((4, 7), (20, 30)), {'b': 3, 'd': 4})
        return g0, g1, g2, g3
    
    @pytest.fixture
    def dict_genes(self, genes) -> dict:
        return {'g1': genes[1].to_dict(), 'g2': genes[2].to_dict(), 'g3': genes[3].to_dict()}
    
    @pytest.fixture
    def json_genes(self, dict_genes) -> dict:
        return json.dumps(dict_genes)
    
    @pytest.mark.xfail(strict=True)
    def test_repeated_genes_chromesome(self, genes):
        chromesome = Chromesome(genes)
    
    def test_chromesome(self, genes):
        chromesome = Chromesome(genes[1:])
        hash(chromesome)
        
    def test_chromesome_from_list(self, genes):
        chromesome = Chromesome(list(genes[1:]))
        hash(chromesome)
    
    def test_chromesome_from_dict(self, dict_genes):
        chromesome = Chromesome.from_dict(dict_genes)
        hash(chromesome)
        
    def test_chromesome_from_json(self, json_genes):
        chromesome = Chromesome.from_json(json_genes)
        hash(chromesome)