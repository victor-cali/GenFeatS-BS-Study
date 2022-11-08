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
        return GenotypeBuilder(nucleobases, chromesome_size=3)
    
    @pytest.fixture
    def gene(self) -> tuple[Gene]:
        return Gene('kurtosis', ['Cz', 'C3'], [[4,12]], {})
    
    def test_instance_GenotypeBuilder(self, nucleobases):
        genobuilder = GenotypeBuilder(nucleobases, chromesome_size=3)
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
        for chromesome in population:
            assert isinstance(chromesome, Chromesome)
    
    def test_mutate_gene(self, genobuilder, gene):
        mutated_gene = genobuilder.mutate_gene(gene)
        assert isinstance(mutated_gene, Gene)
        assert gene.feature == mutated_gene.feature
        assert gene.channels != mutated_gene.channels
        assert gene.frequency_bands != mutated_gene.frequency_bands
        assert gene.feature_parameters == mutated_gene.feature_parameters
        