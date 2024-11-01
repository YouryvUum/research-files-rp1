import logging
from papyrus_scripts.download import download_papyrus
import pandas as pd
from chembl_webresource_client.new_client import new_client
from papyrus_scripts.reader import read_papyrus
import logging
from ast import literal_eval
from papyrus_scripts.preprocess import keep_accession
import pandas as pd



# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("drug_retrieval.log", mode='w')  # Optional: Log to a file
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)


def retrieve_approved_drugs():
    """Retrieve data for all molecules with max_phase=4"""

    molecule = new_client.molecule
    approved_drugs = molecule.filter(max_phase=4).only(
        [
            "molecule_chembl_id",
            "atc_classifications",
            "molecule_type",
            "prodrug",
            "therapeutic_flag",
            "withdrawn_flag",
            "parenteral",
            "oral",
            "topical",
            "structure_type",
            "inorganic_flag",
            "molecule_hierarchy",
            "first_approval",
            "molecule_structures",
            "pref_name",
            "usan_stem",
            "usan_stem_definition",
            "usan_substem",
            "usan_year",
        ]
    )
    logger.info(f"Number of approved drugs: {len(approved_drugs)}")
    approved_drugs = pd.DataFrame(approved_drugs)
    approved_drugs["parent_chembl_id"] = approved_drugs["molecule_hierarchy"].apply(
        lambda x: x["parent_chembl_id"] if x not in [None, {}] else None
    )
    approved_drugs["canonical_smiles"] = approved_drugs["molecule_structures"].apply(
        lambda x: x["canonical_smiles"] if x not in [None, {}] else None
    )
    approved_drugs["standard_inchi"] = approved_drugs["molecule_structures"].apply(
        lambda x: x["standard_inchi"] if x not in [None, {}] else None
    )
    approved_drugs["standard_inchi_key"] = approved_drugs["molecule_structures"].apply(
        lambda x: x["standard_inchi_key"] if x not in [None, {}] else None
    )
    approved_drugs.drop(
        columns=["molecule_structures", "molecule_hierarchy"], inplace=True
    )

    return approved_drugs


def retrieve_mechanisms(approved_drugs_ids: list):
    """Retrieve mechanism(s) of action for each drug

    Args:
        output_dir (str): Path to the output directory
        approved_drugs_ids (list): List of approved drugs molecule ChEMBL IDs
    """
    mechanism = new_client.mechanism
    mechanisms = mechanism.filter(molecule_chembl_id__in=approved_drugs_ids).only(
        [
            "action_type",
            "mechanism_of_action",
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "target_chembl_id",
        ]
    )
    logger.info(f"Number of mechanisms: {len(mechanisms)}")
    return pd.DataFrame(mechanisms)


def retrieve_targets(target_chembl_ids: list):
    """Retrieve target(s) involved in mechanism(s) of action

    Args:
        output_dir (str): Path to the output directory
        target_chembl_ids (list): List of target ChEMBL IDs
    """
    target = new_client.target
    targets = target.filter(target_chembl_id__in=target_chembl_ids).only(
        [
            "organism",
            "target_type",
            "pref_name",
            "target_chembl_id",
            "target_components",
        ]
    )
    targets = pd.DataFrame(targets)
    targets["component_ids"] = targets["target_components"].apply(
        lambda x: [component["component_id"] for component in x]
    )
    targets.drop(columns=["target_components"], inplace=True)
    logger.info(f"Number of targets: {len(targets)}")
    return targets


def retrieve_target_components(target_components_ids: list):
    """Retrieve target components for the specified component ids

    Args:
        output_dir (str): Path to the output directory
        target_components_ids (list): List of target components ChEMBL IDs
    """
    target_component = new_client.target_component
    target_components = target_component.filter(
        component_id__in=target_components_ids
    ).only(["component_id", "protein_classifications", "accession", "description"])
    logger.info(f"Number of target components: {len(target_components)}")
    target_components = pd.DataFrame(target_components)

    target_components["protein_classifications"] = target_components[
        "protein_classifications"
    ].apply(
        lambda x: [classification["protein_classification_id"] for classification in x]
    )

    return target_components


def retrieve_protein_classes():
    """Retrieve all protein classes"""
    protein_class = new_client.protein_classification
    protein_classes = protein_class.only(
        [
            "pref_name",
            "protein_class_id",
            "parent_id",
            "class_level",
        ]
    )
    logger.info(f"Number of protein classes: {len(protein_classes)}")
    return pd.DataFrame(protein_classes)


def fetch_ChEMBL_data(output_dir: str):
    """Fetch approved drugs data from ChEMBL

    This function fetches approved drugs, their mechanisms of action,
    targets, target components, and protein classes from ChEMBL
    and saves the data to CSV files.

    Args:
        output_dir (str): Path to the output directory
    """
    # Retrieve all compounds with max_phase=4
    approved_drugs = retrieve_approved_drugs()
    approved_drugs.to_csv(f"{output_dir}/approved_drugs_chembl.csv", index=False)

    # Retrieve mechanism(s) of action for each drug
    mechanisms = retrieve_mechanisms(approved_drugs["molecule_chembl_id"].tolist())
    mechanisms.to_csv(f"{output_dir}/mechanisms_chembl.csv", index=False)

    # Retrieve target(s) involved in mechanism(s) of action
    targets = retrieve_targets(mechanisms["target_chembl_id"].tolist())
    targets.to_csv(f"{output_dir}/targets_chembl.csv", index=False)

    # Retrieve all target components for each target
    target_components_ids = targets["component_ids"].explode().unique()
    target_components_ids = (
        target_components_ids[~pd.isna(target_components_ids)].astype(int).tolist()
    )
    target_components = retrieve_target_components(target_components_ids)
    target_components.to_csv(f"{output_dir}/target_components_chembl.csv", index=False)

    # Retrieve all protein classes
    protein_classes = retrieve_protein_classes()
    protein_classes.to_csv(f"{output_dir}/protein_classes_chembl.csv", index=False)

def filter_molecules(molecules: pd.DataFrame) -> pd.DataFrame:
    """Filter approved drugs dataset from ChEMBL

    Removes molecules that are not parent molecules, not therapeutics,
    not small molecules, pro-drugs, withdrawn molecules, not oral or
    parenteral drugs, unknown molecule structures, and inorganic molecules

    Args:
        molecules (pd.DataFrame): DataFrame containing molecules data
    """

    total_number_of_mols = len(molecules)
    logger.info(f"Total number of molecules before filtering: {len(molecules)}")

    # Get all unique parent ids in the dataset
    unique_parent_id = molecules["parent_chembl_id"].dropna().unique()
    logger.info(f"    Number of unique parent molecules: {len(unique_parent_id)}")
    logger.info(
        f"    Number of molecules with molecule id is parent id: "
        f"{len(molecules[molecules['molecule_chembl_id'] == molecules['parent_chembl_id']])}"
    )

    # Get molecules with no parent id
    missing_parent_id = molecules[molecules["parent_chembl_id"].isnull()][
        "molecule_chembl_id"
    ].unique()
    logger.info(
        f"    Number of molecules with missing parent id: {len(missing_parent_id)}"
    )

    # Get molecules whose parent is not in the dataset
    no_parent_in_dataset = molecules[
        (molecules["parent_chembl_id"].notnull())
        & (~molecules["parent_chembl_id"].isin(molecules["molecule_chembl_id"]))
    ]["molecule_chembl_id"].unique()

    logger.info(
        f"    Number of molecules whose parent is not in the dataset: {len(no_parent_in_dataset)}"
    )
    logger.info(f"    Ids: {no_parent_in_dataset}")

    # keep only parent molecules (parent_chembl_id == molecule_chembl_id) and
    # molecules with missing parent_chembl_id or whose parent is not in the dataset
    molecules = molecules[
        (molecules["molecule_chembl_id"].isin(unique_parent_id))
        | (molecules["molecule_chembl_id"].isin(missing_parent_id))
        | (molecules["molecule_chembl_id"].isin(no_parent_in_dataset))
    ]

    logger.info(
        f"Number of non-parent molecules dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )

    # keep only therapeutics (as opposed to e.g., an imaging agent, additive etc)
    molecules = molecules[molecules["therapeutic_flag"] == 1]
    logger.info(
        f"Number of non-therapeutic molecules dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # keep only small molecules (molecule_type == "Small molecule")
    molecules = molecules[molecules["molecule_type"] == "Small molecule"]
    logger.info(
        f"Number of non-small molecules dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # Drop pro-drugs (prodrug == 0)
    molecules = molecules[molecules["prodrug"] == 0]
    logger.info(
        f"Number of prodrugs dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # Drop withdrawn molecules (withdrawn_flag == 0)
    molecules = molecules[molecules["withdrawn_flag"] == 0]
    logger.info(
        f"Number of withdrawn molecules dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # Keep only oral or parenteral drugs (oral == 1 or parenteral == 1)
    molecules = molecules[(molecules["oral"] == 1) | (molecules["parenteral"] == 1)]
    logger.info(
        f"Number of non-oral or non-parenteral drugs dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # Keep only known molecule structures (structure_type == "MOL")
    molecules = molecules[molecules["structure_type"] == "MOL"]
    logger.info(
        f"Number of unknown molecule structures dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)

    # keep only organic molecules (inorganic_flag == 0)
    molecules = molecules[molecules["inorganic_flag"] == 0]
    logger.info(
        f"Number of inorganic molecules dropped: "
        f"{total_number_of_mols - len(molecules)}, left: {len(molecules)}"
    )
    total_number_of_mols = len(molecules)
    return molecules

def filter_targets(targets: pd.DataFrame):
    """Filter targets dataset from ChEMBL

    Removes targets that are not single protein targets or not human proteins

    Args:
        targets (pd.DataFrame): DataFrame containing targets data
    """
    # keep only single protein targets (target_type == "SINGLE PROTEIN")
    total_number_of_targets = len(targets)
    logger.info(f"Total number of targets: {len(targets)}")
    targets = targets[targets["target_type"] == "SINGLE PROTEIN"]
    logger.info(
        f"Number of non-single protein targets dropped: "
        f"{total_number_of_targets - len(targets)}, left: {len(targets)}"
    )
    total_number_of_targets = len(targets)

    # keep only human proteins
    targets = targets[targets["organism"] == "Homo sapiens"]
    logger.info(
        f"Number of non-human protein targets dropped: "
        f"{total_number_of_targets - len(targets)}, left {len(targets)}"
    )

    return targets

def filter_mechanisms(
    mechanisms: pd.DataFrame, molecules: pd.DataFrame, targets: pd.DataFrame
):
    """Filter mechanisms of action dataset from ChEMBL

    Removes mechanisms that are not associated with molecules or targets after
    filtering, and targets that are not associated with mechanisms in mechanisms

    Args:
        mechanisms (pd.DataFrame): DataFrame containing mechanisms of action data
        molecules (pd.DataFrame): DataFrame containing filtered molecules data
        targets (pd.DataFrame): DataFrame containing targets data
    """
    # drop mechanisms that are not associated with molecules in molecules
    total_number_of_mechs = len(mechanisms)
    logger.info(f"Total number of mechanisms: {len(mechanisms)}")
    print(mechanisms.columns)
    mechanisms = mechanisms[
        ["action_type", "mechanism_of_action", "molecule_chembl_id", "target_chembl_id"]
    ].merge(
        molecules[
            [
                "molecule_chembl_id",
                "atc_classifications",
                "pref_name",
                "canonical_smiles",
                "standard_inchi",
                "standard_inchi_key",
                "parent_chembl_id",
            ]
        ].set_index("molecule_chembl_id"),
        on="molecule_chembl_id",
        how="inner",
    )
    logger.info(
        f"Number of mechanisms dropped after filtering molecules: "
        f"{total_number_of_mechs - len(mechanisms)}, left: {len(mechanisms)}"
    )

    total_number_of_mechs = len(mechanisms)
    # drop mechanisms that are not associated with targets in targets
    mechanisms = mechanisms.merge(
        targets[
            [
                "organism",
                "target_type",
                "pref_name",
                "target_chembl_id",
                "component_ids",
            ]
        ].set_index("target_chembl_id"),
        on="target_chembl_id",
        how="inner",
        suffixes=("_molecule", "_target"),
    )
    logger.info(
        f"Number of mechanisms dropped after filtering targets: "
        f"{total_number_of_mechs - len(mechanisms)}, left: {len(mechanisms)}"
    )

    logger.info(
        f"Number of unique molecules: {len(mechanisms['molecule_chembl_id'].unique())}"
    )
    logger.info(
        f"Number of unique targets: {len(mechanisms['target_chembl_id'].unique())}"
    )

    return mechanisms

def get_target_classifications(mechanisms: pd.DataFrame, components: pd.DataFrame):
    """Get single protein target component classifications

    Args:
        mechanisms (pd.DataFrame): DataFrame containing mechanisms of action data
        components (pd.DataFrame): DataFrame containing target components data
        output_dir (str): Path to the output directory
    """

    # convert string to list of component ids
    # (should be only one component, as we filtered for single protein targets)
    mechanisms["component_id"] = [
        literal_eval(components)[0] for components in mechanisms["component_ids"]
    ]
    mechanisms.drop(columns=["component_ids"], inplace=True)
    # merge with target components to associate protein classifications
    mechanisms = mechanisms.merge(
        components[["component_id", "protein_classifications", "accession"]],
        on="component_id",
        how="left",
    )

    return mechanisms

def filter_chembl_data(chembl_dir, output_dir: str):
    """Filter ChEMBL data

    Filter molecules, mechanisms, and targets datasets from ChEMBL and
    combine them to get a filtered dataset of approved drugs with their mechanisms
    of action and target information

    Args:
        chembl_dir (str): Path to the ChEMBL data directory, containing the following files:
            - approved_drugs_chembl.csv
            - mechanisms_chembl.csv
            - targets_chembl.csv
            - target_components_chembl.csv
            - protein_classes_chembl.csv
            generated by s01_fetch_data_chembl.py
        output_dir (str): Path to the output directory
    """
    # Filter molecules
    molecules = pd.read_csv(f"{chembl_dir}/approved_drugs_chembl.csv")
    molecules_filtered = filter_molecules(molecules)

    # Add target information to mechanisms and filter based on target type and organism
    targets = pd.read_csv(f"{chembl_dir}/targets_chembl.csv")
    targets_filtered = filter_targets(targets)

    # Drop mechanisms that are not associated with filtered molecules or targets
    mechanisms = pd.read_csv(f"{chembl_dir}/mechanisms_chembl.csv")
    mechanisms_filtered = filter_mechanisms(
        mechanisms, molecules_filtered, targets_filtered
    )

    # Add target component classifications
    components = pd.read_csv(f"{chembl_dir}/target_components_chembl.csv")
    mechanisms_filtered = get_target_classifications(mechanisms_filtered, components)
    mechanisms_filtered.to_csv(
        f"{output_dir}/approved_drugs_mechanisms.csv", index=False
    )

    return mechanisms_filtered

# Fetch data from ChEMBL:
# approved drugs, mechanisms, targets, target components, and protein classes
fetch_ChEMBL_data("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/raw_data_dir/chembl")

# Filter chembl data
filter_chembl_data("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/raw_data_dir/chembl", "/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/chembl_filtered")

# Download the latest version of Papyrus
download_papyrus(version="latest", only_pp=True, descriptors=None, outdir="/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/raw_data_dir/papyrus")

# Load in approved drugs dataset
approved_drugs = pd.read_csv("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/chembl_filtered/approved_drugs_mechanisms.csv")

# Read the papyrus data
papyrus_data = read_papyrus(is3d=False, source_path=None)

papyrus_data = keep_accession(papyrus_data, approved_drugs.accession.unique())

papyrus_data.to_csv("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/papyrus_filtered/papyrus_activities.csv", index=False)
