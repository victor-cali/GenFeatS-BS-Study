import pytest
import numpy as np
from src.genfeats.mapper import Mapper
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome

class TestMapper:
    
    @pytest.fixture
    def mapper(self) -> Mapper:
        return Mapper('D:/Dev/GenFeatS-BS-Study/resources/features.py')
    
    @pytest.fixture
    def genes(self) -> tuple[Gene]:
        g1 = Gene('f1', ((4, 7), (12, 20)), ('A', 'B'), {'a': 1, 'b': 2})
        g2 = Gene('f2', ((4, 7), (15, 20)), ('A', 'C'), {'a': 3, 'c': 4})
        g3 = Gene('f2', ((4, 7), (20, 30)), ('B', 'C'), {'b': 3, 'd': 4})
        return g1, g2, g3
    
    @pytest.fixture
    def chromesome(self, genes: tuple[Gene]) -> Chromesome:
        return Chromesome(genes)
    
    def test_map_gene(self, mapper: Mapper, genes: tuple[Gene]) -> None:
        gene = genes[0]
        phenotype = mapper.to_phenotype(gene)
        assert isinstance(phenotype, np.ndarray)
        assert np.isfinite(phenotype).all()
    
    def test_map_chromesome(self, mapper: Mapper, chromesome: Chromesome) -> None:
        phenotype = mapper.to_phenotype(chromesome)
        assert isinstance(phenotype, np.ndarray)
        assert np.isfinite(phenotype).all()