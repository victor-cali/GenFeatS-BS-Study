import os
import sys

from sklearn import svm

from src.genfeats.genfeats_bs import GenFeatSBS
from src.sighandling.sighandling import get_dataset_bbcic4_2b

DATASET_FOLDER_PATH = os.getenv('DATASET_FOLDER_PATH') #'D:/Dev/GenFeatS-BS/data/external/segmented' #os.getenv('DATASET_FOLDER_PATH')
RESOURCES_FOLDER_PATH = os.getenv('RESOURCES_FOLDER_PATH') #'D:/Dev/GenFeatS-BS/resources' #os.getenv('RESOURCES_FOLDER_PATH')
RESULTS_FOLDER_PATH = os.getenv('RESULTS_FOLDER_PATH') #'D:/Dev/GenFeatS-BS/data/processed' #os.getenv('RESULTS_FOLDER_PATH')
SUBJECT = str(sys.argv[1])

def get_base_name(file_name: str) -> str: 
    base_name, _ = os.path.splitext(file_name)
    return base_name

if __name__ == '__main__':
     
    data_file = f'{DATASET_FOLDER_PATH}/{SUBJECT}'

    epochs = get_dataset_bbcic4_2b(data_file)

    resources_folder = RESOURCES_FOLDER_PATH
    
    fitness_function = svm.SVC(kernel='rbf')
    
    results_path = f'{RESULTS_FOLDER_PATH}/{get_base_name(SUBJECT)}/'
    
    metadata = {
        'subject': get_base_name(SUBJECT),
        'survival_rate': 0.1,
        'chromesome_size': 5,
        'population_size': 50,
        'extintions_limit': 10,
        'generations_limit': 1000
    }
    genfeatsbs = GenFeatSBS(
        resources_folder, 
        epochs, 
        fitness_function, 
        results_path=results_path, 
        **metadata 
    )
    result = genfeatsbs()