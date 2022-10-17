import pytest
import numpy as np
from src.genfeats.mapper import Mapper
from src.genfeats.dna.gene import Gene
from src.genfeats.dna.chromesome import Chromesome
from src.sighandling.sighandling import get_dataset_bbcic4_2b


class TestMapper:
    
    @pytest.fixture
    def epochs(self):
        new_names = dict()
        filtered_epochs = list()
        epochs = get_dataset_bbcic4_2b('D:/Dev/GenFeatS-BS-Study/data/external/segmented/S4_clean.mat')
        for name in epochs.ch_names:
            new_names.update({name: f'{name}(4-12)'})
        subepochs = epochs.copy()
        subepochs.filter(4, 12, method = 'iir', verbose = 50)
        subepochs.rename_channels(new_names)
        filtered_epochs.append(subepochs.copy())
        epochs.add_channels(filtered_epochs, force_update_info=True)
        return epochs
    
    @pytest.fixture
    def mapper(self, epochs) -> Mapper:
        features_file = 'D:/Dev/GenFeatS-BS-Study/resources/gfsbs_features.py'
        return Mapper(features_file=features_file, chromesome_size=3, epochs=epochs)
    
    @pytest.fixture
    def genes(self) -> tuple[Gene]:
        g1 = Gene('kurtosis', [[4,12]], ['Cz', 'C3'], {})
        g2 = Gene('variance', [[4,12]], ['Cz'], {})
        g3 = Gene('skewness', [[4,12]], ['Cz'], {})
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