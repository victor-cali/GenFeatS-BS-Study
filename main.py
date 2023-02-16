from sklearn import svm
from src.genfeats.genfeats_bs import GenFeatSBS
from src.sighandling.sighandling import get_dataset_bbcic4_2b

subjects = {
    'S1': './data/external/segmented/S1_clean.mat',
    'S2': './data/external/segmented/S2_clean.mat',
    'S3': './data/external/segmented/S3_clean.mat',
    'S4': './data/external/segmented/S4_clean.mat',
    'S5': './data/external/segmented/S5_clean.mat',
    'S6': './data/external/segmented/S6_clean.mat',
    'S7': './data/external/segmented/S7_clean.mat',
    'S8': './data/external/segmented/S8_clean.mat',
    'S9': './data/external/segmented/S9_clean.mat'
}

resources_folder = './resources/'
fitness_function = svm.SVC(kernel='rbf')
results_path = 'D:/Dev/GenFeatS-BS-Study/data/processed/S2/'

for subject, path in subjects.items():
    for _ in range(10):
        
        execution_metadata = {
            'subject': subject,
            'survival_rate': 0.1,
            'chromesome_size': 5,
            'population_size': 50,
            'extintions_limit': 10,
            'generations_limit': 1000
        }
        
        epochs = get_dataset_bbcic4_2b(path)
        
        genfeatsbs = GenFeatSBS(
            resources_folder, 
            epochs, 
            fitness_function, 
            results_path=results_path, 
            execution_metadata=execution_metadata,
            chromesome_size=5,
            population_size=50,
            extintions_limit=10,
            generations_limit=1000
        )
        
        result = genfeatsbs()