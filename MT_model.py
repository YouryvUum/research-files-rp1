import os
import numpy as np
from qsprpred.extra.gpu.models.chemprop import ChempropModel
from qsprpred.models.early_stopping import EarlyStoppingMode
from qsprpred.models import CrossValAssessor, TestSetAssessor
from qsprpred.models.metrics import MaskedRMSE
import logging
from qsprpred.data import QSPRDataset
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "/home/youry23/rp1/Approved_drugs_Chemprop/A/model/MT_all_epoch=200_random1"
maskedrmse = MaskedRMSE()

# Load dataset
dataset = QSPRDataset.fromFile("/home/youry23/rp1/Approved_drugs_Chemprop/A/data_management/processed_data_dir/dataset/MT_unapp_dataset_random/MT_unapp_dataset_random_meta.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_assess_model(base_dir, model_name, parameters, dataset):
    try:
        # Suppress all FutureWarnings within this function
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize the Chemprop model
        model = ChempropModel(
            base_dir=base_dir,
            name=model_name,
            parameters=parameters,
            quiet_logger=False,
            random_state=22
        )
        
        # Reset early stopping epochs
        model.earlyStopping._trainedEpochs = []
        
        # Cross-validation assessment
        logging.info("Starting cross-validation...")
        CrossValAssessor(maskedrmse, mode=EarlyStoppingMode.RECORDING)(model, dataset)
        
        # Test set assessment
        logging.info("Starting test set assessment...")
        TestSetAssessor(maskedrmse, mode=EarlyStoppingMode.RECORDING)(model, dataset)
        
        # Print trained epochs
        print(f"Trained epochs: {model.earlyStopping.trainedEpochs}")
        
        # Set aggregation function and fit the model using optimal early stopping
        model.earlyStopping.aggregateFunc = np.mean
        model.fitDataset(dataset, mode=EarlyStoppingMode.OPTIMAL)

    except Exception as e:
        logging.error(f"Error during model training and assessment: {e}")
        raise

if __name__ == "__main__":
    # Train and assess model
    train_and_assess_model(base_dir=OUTPUT_DIR, model_name="MT_all_epoch=200_random1", parameters={"epochs": 200}, dataset=dataset)
