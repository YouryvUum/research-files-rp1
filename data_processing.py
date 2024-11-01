import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor, as_completed

PROCESSED_DATA_DIR = "/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/papyrus_filtered"

# Load in approved drugs dataset
approved_drugs = pd.read_csv("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/chembl_filtered/approved_drugs_mechanisms.csv")

# Load in activities dataset
papyrus_activities = pd.read_csv("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/papyrus_filtered/papyrus_activities.csv")

# Define a function to process each accession
def process_accession(ACCESSION):
    print(f"Processing {ACCESSION}...")
    papyrus_activities_subset = papyrus_activities[papyrus_activities["accession"] == ACCESSION]

    # Load in mmp dataset
    try:
        mmp = pd.read_csv(
            f"{PROCESSED_DATA_DIR}/mmpdb/index/{ACCESSION}.csv", sep="\t", header=None
        )
    except pd.errors.EmptyDataError:
        print(f"No mmp data for {ACCESSION}")
        return ACCESSION  # Append to no_data_list
    except FileNotFoundError:
        print(f"No file found for {ACCESSION}")
        return ACCESSION

    mmp.columns = ["Molecule1", "Molecule2", "InchI", "InchI", "Transformation", "Core"]

    # Get Morgan fingerprints for each molecule in mmp dataset and papyrus dataset
    papyrus_mols = papyrus_activities_subset["SMILES"].apply(
        lambda x: Chem.MolFromSmiles(x)
    )
    papyrus_fps = papyrus_mols.apply(lambda x: AllChem.GetMorganFingerprint(x, 2))

    approved_drugs_smiles = approved_drugs[approved_drugs["accession"] == ACCESSION][
        "canonical_smiles"
    ].reset_index(drop=True)
    approved_drugs_mols = approved_drugs_smiles.apply(lambda x: Chem.MolFromSmiles(x))
    approved_drugs_mols.apply(lambda x: Chem.RemoveStereochemistry(x))
    approved_drugs_mols_fps = approved_drugs_mols.apply(
        lambda x: AllChem.GetMorganFingerprint(x, 2)
    ).reset_index(drop=True)

    # Find most similar SMILES to approved drugs smiles in activities
    sims = []
    for i in range(len(approved_drugs_mols_fps)):
        sims.append(
            DataStructs.BulkTanimotoSimilarity(
                approved_drugs_mols_fps[i], list(papyrus_fps)
            )
        )

    sims = pd.DataFrame(sims)
    sims.columns = papyrus_activities_subset["SMILES"]
    sims.index = approved_drugs_smiles
    # For each row, find the column with the highest similarity
    sims["similarity"] = sims.apply(lambda x: x.max(), axis=1)
    sims["most_similar_smiles"] = sims.drop("similarity", axis=1).apply(
        lambda x: sims.columns[x.argmax()], axis=1
    )

    os.makedirs(f"{PROCESSED_DATA_DIR}/most_similar_in_papyrus", exist_ok=True)
    sims[["similarity", "most_similar_smiles"]].to_csv(
        f"{PROCESSED_DATA_DIR}/most_similar_in_papyrus/{ACCESSION}.csv"
    )

    # Drop if similarity is less than 0.9
    sims = sims[sims["similarity"] > 0.9]

    # Mark approved drugs in papyrus dataset
    papyrus_activities.loc[
        papyrus_activities["accession"] == ACCESSION, "approved_drug"
    ] = papyrus_activities["SMILES"].isin(sims["most_similar_smiles"])

    return None  # No errors

# Use ThreadPoolExecutor to run in parallel
no_data_list = []
accessions = papyrus_activities["accession"].unique()

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_accession, accession) for accession in accessions]
    for future in as_completed(futures):
        result = future.result()
        if result:  # If there's an issue (accession with no data)
            no_data_list.append(result)

# Drop Unnamed column
papyrus_activities = papyrus_activities.drop(columns="Unnamed: 0")

# Drop rows where "source" column contains "Christmann2016"
papyrus_activities = papyrus_activities[~papyrus_activities["source"].str.contains("Christmann2016", na=False)]

# Reset the index
papyrus_activities.reset_index(drop=True, inplace=True)

# Save updated papyrus_activities and no_data_list
papyrus_activities.to_csv(
    f"{PROCESSED_DATA_DIR}/papyrus_activities_updated.csv", index=False
)

print("papyrus_activities_updated saved!")

pd.DataFrame(no_data_list).to_csv(f"{PROCESSED_DATA_DIR}/most_similar_in_papyrus/no_data_list.csv", index=False)

pap_all_human = papyrus_activities.copy()

# remove all unwanted columns
columns_to_keep = ['SMILES', 'target_id', 'pchembl_value_Mean', 'approved_drug']
pap_all_human = pap_all_human[columns_to_keep]
print("Removing columns succesfull")

# Pivot the DataFrame
all_human = pd.pivot_table(pap_all_human, values='pchembl_value_Mean', index='SMILES', columns='target_id', aggfunc='first')

# Merge the 'Approved' column to the pivoted DataFrame
all_human = all_human.merge(pap_all_human[['SMILES', 'approved_drug']].drop_duplicates(), on='SMILES')

# Create a copy of the DataFrame excluding the specified columns
columns_to_exclude = ['SMILES', 'approved_drug']
df_filtered = all_human.drop(columns=columns_to_exclude)

# Count the non-null values for each column
counts = df_filtered.count()

# Create a new DataFrame for the counts
counts_df = pd.DataFrame(counts).transpose()

# Add the columns to exclude with NaN values (or another placeholder if desired)
for col in columns_to_exclude:
    counts_df[col] = float('nan')

# Reorder columns to match the original DataFrame
counts_df = counts_df[all_human.columns]

# Append the counts row to the original DataFrame using concat
all_human = pd.concat([all_human, counts_df], ignore_index=True)

# Save approved dataframe
app_human = all_human[all_human['approved_drug'] == True].copy()

pap_app = app_human.copy()

# Keep "SMILES" and "approved_drug" intact when filtering columns
columns_to_keep = ["SMILES", "approved_drug"]

# Step 1: Remove columns (except "SMILES" and "approved_drug") where all values are NaN
df_filtered_columns = pap_app.loc[:, pap_app.columns.isin(columns_to_keep) | pap_app.notna().any()]

# Step 2: Remove rows where all values (except in "SMILES" and "approved_drug") are NaN
df_filtered_rows = df_filtered_columns.dropna(axis=0, how="all", subset=[col for col in df_filtered_columns.columns if col not in columns_to_keep])

# Step 3: Reset the index (optional)
df_filtered_rows.reset_index(drop=True, inplace=True)

# The resulting DataFrame after filtering
app_human_filtered = df_filtered_rows

app_human.to_csv(f"{PROCESSED_DATA_DIR}/papyrus_activities_approved.csv", index=False)

# Get the counts from the last row
counts_row = all_human.iloc[-1]
counts_row

# Filter the columns based on counts greater than 30 and excluding specified columns
selected_columns = counts_row[counts_row > 30].index.difference(['SMILES', 'approved_drug'])

# Filter the DataFrame to keep only the selected columns
all_human_ge_30 = all_human[selected_columns]

# Include original columns SMILES, approved_drug
all_human_ge_30 = pd.concat([all_human[['SMILES', 'approved_drug']], all_human_ge_30], axis=1)

# Assuming `all_human_ge_30` is your DataFrame
df = all_human_ge_30.copy()

# Step 1: Identify the accession columns
accession_columns = df.columns.difference(['SMILES', 'approved_drug'])

# Step 2: Remove columns with all missing values in the accession columns
removed_columns = df[accession_columns].columns[df[accession_columns].isna().all()].tolist()
num_removed_columns = len(removed_columns)
df.drop(columns=removed_columns, inplace=True)

# Step 3: Remove rows with all missing values in the remaining accession columns
removed_rows = df[df[accession_columns].isna().all(axis=1)]['SMILES'].tolist()
num_removed_rows = len(removed_rows)
df = df.dropna(subset=accession_columns, how='all')

# Logging removed column headers and SMILES values
print(f"Removed columns: {removed_columns}")
print(f"Number of columns removed: {num_removed_columns}")
print(f"Removed SMILES values: {removed_rows}")
print(f"Number of SMILES rows removed: {num_removed_rows}")
all_human_ge_30 = df

all_human_ge_30_save = all_human_ge_30[:-1]

# Save unapproved dataframe (greater than or equal to 30 datapoints per target)
unapp_human = all_human_ge_30_save[all_human_ge_30_save['approved_drug'] == False].copy()
unapp_human.to_csv(f"{PROCESSED_DATA_DIR}/papyrus_activities_unapproved_ge_30.csv", index=False)