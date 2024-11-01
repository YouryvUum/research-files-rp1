import pandas as pd
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import pandas as pd
from qsprpred.logs import logger
from qsprpred.logs import setLogger
import logging
from qsprpred.data import QSPRDataset
from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.data.descriptors.fingerprints import MorganFP
import pandas as pd
from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.data.descriptors.fingerprints import MorganFP
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from qsprpred.data.sampling.splits import DataSplit
from typing import Iterable
import numpy as np
PROCESSED_DATA_DIR = "/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/papyrus_filtered"
DATASET_DIR = "/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/dataset2"

logger.setLevel(logging.DEBUG)
setLogger(logger)

def process_smiles_with_progress(smiles_list):
    results = []
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold.UpdatePropertyCache()
            results.append(True)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            results.append(False)
    return results

class MySplit(DataSplit):
    """My customized split"""

    def __init__(self, test_ids: list[list[str]]):
        super().__init__()
        self.test_ids = test_ids

    def split(
        self,
        X: np.ndarray | pd.DataFrame, 
        y: np.ndarray | pd.DataFrame | pd.Series
    ) -> Iterable[tuple[list[int], list[int]]]:
        """Uses only the specified IDs from the data set as test set
        Returns an iterator of training and test split indices, 
        just like a scikit learn splitter would.
        """
        splits = []
        for test_ids in self.test_ids:
            test = np.where(X.index.isin(test_ids))[0]
            train = np.where(~X.index.isin(test_ids))[0]
            splits.append((train, test))
        return splits

# Load unapproved dataset
unapp_human_ge_30 = pd.read_csv("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/papyrus_filtered/papyrus_activities_unapproved_ge_30.csv")

# Create list of SMILES
smiles_list = unapp_human_ge_30['SMILES'].tolist()
unapp_human_ge_30['is_processed'] = process_smiles_with_progress(smiles_list)

# Remove invalid SMILES from dataframe
unapp_human_ge_30 = unapp_human_ge_30[unapp_human_ge_30["is_processed"]==True]

# Drop the 'is_processed' and 'approved_drug' columns from the DataFrame
unapp_human_ge_30 = unapp_human_ge_30.drop(columns=['is_processed', "approved_drug"])

# Select columns to count, excluding "SMILES"
columns_to_count = [col for col in unapp_human_ge_30.columns if col not in ['SMILES']]

# Calculate the total count of non-NaN values in the selected columns
total_non_nan_count = unapp_human_ge_30[columns_to_count].notna().sum().sum()

print(f"Total count of non-NaN values (excluding 'SMILES'): {total_non_nan_count}")

# Calculate the count of non-NaN values in the selected columns for each row
unapp_human_ge_30['NumDataPoints'] = unapp_human_ge_30[columns_to_count].notna().sum(axis=1)

# Calculate the total number of data points
total_data_points = unapp_human_ge_30['NumDataPoints'].sum()

# Calculate the number of data points for train and test sets
num_train_data_points = int(0.9 * total_data_points)
num_test_data_points = total_data_points - num_train_data_points

# Initialize lists to collect redistributed data
train_rows = []
test_rows = []

# Shuffle SMILES for randomness
unapp_human_ge_30 = unapp_human_ge_30.sample(frac=1, random_state=42).reset_index(drop=True)

# Redistribute SMILES based on the number of data points
current_train_data_points = 0
current_test_data_points = 0

for index, row in unapp_human_ge_30.iterrows():
    num_points = row['NumDataPoints']
    
    if current_train_data_points + num_points <= num_train_data_points:
        # Add row to train set
        train_rows.append(row)
        current_train_data_points += num_points
    else:
        # Add row to test set
        test_rows.append(row)
        current_test_data_points += num_points

# Convert lists to DataFrames
train_df_final = pd.DataFrame(train_rows)
test_df_final = pd.DataFrame(test_rows)

# Reset indices
train_df_final.reset_index(drop=True, inplace=True)
train_df_final = train_df_final.drop(columns="NumDataPoints")
test_df_final.reset_index(drop=True, inplace=True)
test_df_final = test_df_final.drop(columns="NumDataPoints")
unapp_human_ge_30 = unapp_human_ge_30.drop(columns=["NumDataPoints"])

# Create list to find indexes
test_index_list = test_df_final["SMILES"].tolist()

# Find the indexes of rows where the SMILES value is in the test_index_list
matching_indexes = unapp_human_ge_30[unapp_human_ge_30['SMILES'].isin(test_index_list)].index

# Convert to a list
matching_indexes_list = matching_indexes.tolist()

# Select only target columns
columns_to_exclude = ['SMILES']
selected_columns = unapp_human_ge_30.drop(columns=columns_to_exclude)

# Create a list of dictionaries for each accession code
target_props = [{"name": col, "task": "REGRESSION"} for col in selected_columns]

dataset = QSPRDataset(
    name="MT_unapp_dataset2",
    df=unapp_human_ge_30,
    target_props=target_props,
    store_dir=f"{DATASET_DIR}",
    overwrite=True,
    drop_empty=False,
    random_state=42
)

datasetdf = dataset.getDF()

# Find the indexes of rows where the SMILES value is in the test_index_list
matching_indexes = datasetdf[datasetdf['SMILES'].isin(test_index_list)].index

# Convert to a list (optional)
matching_indexes_list = matching_indexes.tolist()

# train-test split
my_split = MySplit([matching_indexes_list])

dataset.nJobs = 20
logger.setLevel(logging.DEBUG)
setLogger(logger)

# calculate compound features and split dataset into train and test
dataset.prepareDataset(
    split=my_split,
    feature_calculators=[SmilesDesc(), MorganFP(radius=3, nBits=2048)]
)

print(f"Number of samples train set: {len(dataset.y)}")
print(f"Number of samples test set: {len(dataset.y_ind)}")

dataset.save()