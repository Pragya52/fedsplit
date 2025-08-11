"""
Configuration file for Federated LLaMA-2 Medical QA
"""

import torch
from pathlib import Path

class Config:
    # Model Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    SPLIT_LAYER = 16  # Split point between client and server
    MAX_LENGTH = 512
    
    # Federated Learning Configuration
    NUM_CLIENTS = 3
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 1
    
    # Training Configuration
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0
    
    # Data Configuration
    MAX_SAMPLES = 10000
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    
    # Privacy Configuration
    PRIVACY_SIGMA = 0.1
    QUANTIZATION_BITS = 8
    
    # Evaluation Configuration
    EVAL_FREQUENCY = 2
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure directories exist
    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset Selection (use only one at a time)
    DATASET_NAME = "pubmed_qa"  # Options: "pubmed_qa", "medqa", "medmcqa"
    
    # Non-IID Distribution Configuration
    NON_IID_TYPE = "specialty"  # Options: "specialty", "difficulty", "question_type", "dirichlet"
    NON_IID_ALPHA = 0.5  # Dirichlet concentration (lower = more non-IID)
    
    @classmethod
    def get_dataset_config(cls):
        """Get configuration for selected dataset"""
        configs = {
            "pubmed_qa": {
                "dataset_path": "pubmed_qa",
                "subset": "pqa_labeled",
                "split": "train",
                "question_key": "question",
                "answer_key": "long_answer",
                "context_key": "context"
            },
            "medqa": {
                "dataset_path": "medqa",
                "subset": "en", 
                "split": "train",
                "question_key": "question",
                "answer_key": "answer",
                "options_key": "options"
            },
            "medmcqa": {
                "dataset_path": "medmcqa",
                "subset": None,
                "split": "train", 
                "question_key": "question",
                "answer_key": "cop",
                "options_keys": ["opa", "opb", "opc", "opd"]
            }
        }
        return configs[cls.DATASET_NAME]
