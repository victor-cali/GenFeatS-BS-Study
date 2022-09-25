import pytest
import json
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome
from src.genfeats.genotype_builder import GenotypeBuilder

class TestGenotypeBuilder:

    @pytest.fixture
    def nucleobases(self) -> str:
        return 'D:/Dev/GenFeatS-BS-Study/resources/nucleobases.json'

    @pytest.fixture
    def genobuilder(self, nucleobases) -> str:
        return GenotypeBuilder(nucleobases, chromesome_len=3)
    
    def test_instance_GenotypeBuilder(self, nucleobases):
        genobuilder = GenotypeBuilder(nucleobases, chromesome_len=3)
        assert isinstance(genobuilder, GenotypeBuilder)

    def test_make_gene(self, genobuilder):
        gene = genobuilder.make_gene()
        assert isinstance(gene, Gene)
    
    def test_make_chromesome(self, genobuilder):
        chromesome = genobuilder.make_chromesome()
        assert isinstance(chromesome, Chromesome)
    
    def test_make_population(self, genobuilder):
        size = 10
        population = genobuilder.make_population(size)
        assert len(population) == size
        assert all(isinstance(population, Chromesome))