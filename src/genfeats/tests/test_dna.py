import pytest
import json
from src.genfeats.dna.gene import Gene

class TestDNAModel:

    @pytest.fixture
    def dict_gene(self) -> dict:
        gene = {
            'func': 'dummy_func',
            'freq': [[4, 7.5], [12, 20]],
            'source': ['C3', 'C4'],
            'func_args': {
                'dummy_arg1': 1,
                'dummy_arg2': 'Hello world'
            } 
        }
        return gene
    
    @pytest.fixture
    def json_gene(self, dict_gene) -> str:
        return json.dumps(dict_gene)
    
    def test_gene_build_by_value(self, dict_gene):
        func = dict_gene['func']
        freq = dict_gene['freq']
        source = dict_gene['source']
        func_args = dict_gene['func_args']
        gene = Gene(func, freq, source, func_args)
        hash(gene)
        
    def test_gene_build_by_dict(self, dict_gene):
        gene = Gene(**dict_gene)
        hash(gene)
        
    def test_gene_build_by_json(self, json_gene):
        gene = Gene(json_gene)
        hash(gene)
