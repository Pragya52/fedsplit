#!/usr/bin/env python3
"""
Main script for Federated LLaMA-2-7B Medical QA
Fixed version with single dataset focus and proper error handling
"""

import argparse
import logging
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config import Config
from federated.federated_trainer import FederatedTrainer

def setup_logging():
    """Setup logging configuration"""
    Config.setup_directories()
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = Config.LOGS_DIR / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def check_requirements():
    """Check system requirements"""
    logger = logging.getLogger(__name__)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! LLaMA-2-7B requires GPU")
        return False
    
    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
    
    if gpu_memory < 12:
        logger.warning(f"⚠️ GPU memory ({gpu_memory:.1f}GB) may be insufficient for LLaMA-2-7B")
        logger.warning("Consider using smaller batch size or model quantization")
    
    return True

def estimate_training_time():
    """Estimate training time"""
    logger = logging.getLogger(__name__)
    
    samples_per_client = Config.MAX_SAMPLES // Config.NUM_CLIENTS
    batches_per_client = samples_per_client // Config.BATCH_SIZE
    total_batches = batches_per_client * Config.NUM_CLIENTS * Config.NUM_ROUNDS
    
    # Estimate ~1-2 seconds per batch for LLaMA-2-7B
    estimated_seconds = total_batches * 1.5
    estimated_hours = estimated_seconds / 3600
    
    logger.info(f"📊 Training Estimation:")
    logger.info(f"  Dataset: {Config.DATASET_NAME}")
    logger.info(f"  Total samples: {Config.MAX_SAMPLES}")
    logger.info(f"  Samples per client: {samples_per_client}")
    logger.info(f"  Total batches: {total_batches}")
    logger.info(f"  Estimated time: {estimated_hours:.1f} hours")
    
    return estimated_hours

def print_configuration():
    """Print current configuration"""
    logger = logging.getLogger(__name__)
    
    logger.info("🦙 FEDERATED LLAMA-2-7B MEDICAL QA")
    logger.info("=" * 60)
    logger.info("📋 Configuration:")
    logger.info(f"  Model: {Config.MODEL_NAME}")
    logger.info(f"  Dataset: {Config.DATASET_NAME}")
    logger.info(f"  Non-IID Type: {Config.NON_IID_TYPE}")
    logger.info(f"  Non-IID Alpha: {Config.NON_IID_ALPHA}")
    logger.info(f"  Split Layer: {Config.SPLIT_LAYER}")
    logger.info(f"  Clients: {Config.NUM_CLIENTS}")
    logger.info(f"  Rounds: {Config.NUM_ROUNDS}")
    logger.info(f"  Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"  Max Samples: {Config.MAX_SAMPLES}")
    logger.info(f"  Learning Rate: {Config.LEARNING_RATE}")
    logger.info(f"  Privacy σ: {Config.PRIVACY_SIGMA}")
    logger.info(f"  Device: {Config.DEVICE}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Federated LLaMA-2-7B Medical QA")
    
    # Model arguments
    parser.add_argument('--model_name', default=Config.MODEL_NAME, 
                       help='LLaMA model name')
    parser.add_argument('--split_layer', type=int, default=Config.SPLIT_LAYER,
                       help='Layer to split model at')
    
    # Federated arguments
    parser.add_argument('--num_clients', type=int, default=Config.NUM_CLIENTS,
                       help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=Config.NUM_ROUNDS,
                       help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=Config.LOCAL_EPOCHS,
                       help='Local epochs per round')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=Config.MAX_SAMPLES,
                       help='Maximum samples to use')
    
    # Dataset arguments
    parser.add_argument('--dataset', choices=['pubmed_qa', 'medqa', 'medmcqa'],
                       default=Config.DATASET_NAME,
                       help='Medical dataset to use')
    
    # Non-IID arguments
    parser.add_argument('--non_iid_type', choices=['specialty', 'difficulty', 'question_type', 'dirichlet'],
                       default=Config.NON_IID_TYPE,
                       help='Type of Non-IID distribution')
    parser.add_argument('--non_iid_alpha', type=float, default=Config.NON_IID_ALPHA,
                       help='Dirichlet concentration parameter (lower = more non-IID)')
    
    # Privacy arguments
    parser.add_argument('--privacy_sigma', type=float, default=Config.PRIVACY_SIGMA,
                       help='Gaussian noise sigma for privacy')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal samples')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Update config with command line arguments
        Config.MODEL_NAME = args.model_name
        Config.SPLIT_LAYER = args.split_layer
        Config.NUM_CLIENTS = args.num_clients
        Config.NUM_ROUNDS = args.num_rounds
        Config.LOCAL_EPOCHS = args.local_epochs
        Config.BATCH_SIZE = args.batch_size
        Config.LEARNING_RATE = args.learning_rate
        Config.MAX_SAMPLES = args.max_samples
        Config.DATASET_NAME = args.dataset
        Config.NON_IID_TYPE = args.non_iid_type
        Config.NON_IID_ALPHA = args.non_iid_alpha
        Config.PRIVACY_SIGMA = args.privacy_sigma
        
        # Quick test mode
        if args.quick_test:
            Config.MAX_SAMPLES = 1000
            Config.NUM_ROUNDS = 3
            Config.BATCH_SIZE = 1
            logger.info("🚀 QUICK TEST MODE ENABLED")
        
        # Print configuration
        print_configuration()
        
        # Check requirements
        if not check_requirements():
            logger.error("❌ System requirements not met")
            return 1
        
        # Estimate training time
        estimated_hours = estimate_training_time()
        
        if estimated_hours > 12 and not args.quick_test:
            response = input(f"\n⚠️ Estimated training time: {estimated_hours:.1f} hours. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Training cancelled by user")
                return 0
        
        # Create and run trainer
        logger.info("\n🚀 Starting federated training...")
        trainer = FederatedTrainer()
        final_results = trainer.train_federated()
        
        # Print final results
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Final Results:")
        logger.info(f"  Best Accuracy: {final_results['best_accuracy']:.4f}")
        logger.info(f"  Total Time: {final_results['total_time_hours']:.2f} hours")
        logger.info(f"  Rounds: {final_results['rounds_completed']}")
        logger.info(f"  Privacy ε: {final_results['privacy_status']['epsilon_spent']:.3f}")
        logger.info(f"  Results saved to: {Config.RESULTS_DIR}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        logger.error("\n💡 Common solutions:")
        logger.error("  1. Request LLaMA-2 access: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        logger.error("  2. Login: huggingface-cli login")
        logger.error("  3. Check GPU memory (need 12GB+)")
        logger.error("  4. Try smaller batch size: --batch_size 1")
        logger.error("  5. Try quick test: --quick_test")
        logger.error("  6. Install requirements: pip install -r requirements.txt")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
