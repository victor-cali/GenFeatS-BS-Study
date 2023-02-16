# %%
import pandas as pd
import json

# %%
def retrieve_data(data_file_path: str) -> dict:
    with open(data_file_path) as data_file:
        data = json.load(data_file)
        return data

# %%
def get_subject(data: dict) -> str:
    return data['info']['metadata']['subject']

# %%
def get_last_generation(data: dict) -> dict:
    if 'info' in data.keys():
        del data['info']
    best_solution = max(map(int, data.keys()))
    return data[str(best_solution)]

# %%
def make_row(subject: str, generation_record: dict) -> dict:
    row = dict()
    row['subject'] = subject
    solution = generation_record['solution']
    for gene, value in solution.items():
        del value['feature_parameters']
        value_keys = tuple(value.keys())
        for key in value_keys:
            value[f'{key}_{gene}'] = value[key]
            del value[key]
    flat_solution = dict()
    for key in solution:
        flat_solution = flat_solution | solution[key]
    del generation_record['solution']
    row = row | flat_solution | generation_record
    return row

# %%
def encode(row: dict) -> dict:
    for key, value in row.items():
        if not isinstance(value, str):
            row.update({key: repr(value)})

# %%
data_file_path = '../data/processed/S2/results-S2-08-01-2023-21-38-51.json'

# %%
data = retrieve_data(data_file_path)

# %%
subject = get_subject(data)

# %%
generation_record = get_last_generation(data)

# %%
row = make_row(subject, generation_record)

# %%
encoded_row = encode(row)

# %%
print(row)


