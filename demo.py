#!/usr/bin/env python3
"""
Demo script for Federated LLaMA-2 Medical QA
Shows different Non-IID distributions and generates sample outputs
"""

import torch
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config import Config
from data.dataset import MedicalQADataset
from data.non_iid_distribution import NonIIDDistributor
from models.llama_models import LLaMAModelManager

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def demo_non_iid_distribution():
    """Demonstrate Non-IID data distribution"""
    logger = setup_logging()
    
    logger.info("🚀 Federated LLaMA-2 Medical QA Demo")
    logger.info("=" * 50)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, demo will be limited")
    
    try:
        # Load a small sample dataset
        logger.info("📊 Loading sample medical dataset...")
        
        # Create a dummy tokenizer for demo
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create small dataset for demo
        Config.MAX_SAMPLES = 1000  # Small for demo
        dataset = MedicalQADataset(tokenizer, max_samples=Config.MAX_SAMPLES)
        
        logger.info(f"✅ Loaded {len(dataset)} samples from {Config.DATASET_NAME}")
        
        # Demo different Non-IID distributions
        distributions = ['specialty', 'difficulty', 'question_type', 'dirichlet']
        alphas = [0.1, 0.5, 1.0]
        
        for dist_type in distributions:
            logger.info(f"\n🔍 Testing {dist_type} distribution:")
            
            for alpha in alphas:
                logger.info(f"\n  Alpha = {alpha} ({'Highly Non-IID' if alpha < 0.5 else 'Moderate' if alpha < 1.0 else 'Nearly IID'}):")
                
                distributor = NonIIDDistributor(
                    dataset=dataset,
                    num_clients=3,
                    distribution_type=dist_type
                )
                
                client_datasets = distributor.create_non_iid_splits(alpha=alpha)
                analysis = distributor.analyze_distribution(client_datasets)
                
                # Show sample questions from each client
                for client_id, client_dataset in enumerate(client_datasets):
                    if len(client_dataset) > 0:
                        sample = client_dataset[0]
                        question_preview = sample['question'][:100] + "..." if len(sample['question']) > 100 else sample['question']
                        logger.info(f"    Client {client_id} ({len(client_dataset)} samples): {question_preview}")
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("\n💡 To run full training:")
        logger.info("   python main.py --quick_test")
        logger.info("   python main.py --dataset pubmed_qa --non_iid_type specialty --non_iid_alpha 0.5")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.info("\n💡 Common solutions:")
        logger.info("  1. Install requirements: pip install -r requirements.txt")
        logger.info("  2. Check internet connection for dataset download")
        logger.info("  3. Try with smaller sample: Config.MAX_SAMPLES = 100")

def demo_model_architecture():
    """Demonstrate model architecture"""
    logger = setup_logging()
    
    logger.info("\n🏗️ Model Architecture Demo")
    logger.info("=" * 50)
    
    try:
        # This would normally load LLaMA-2, but for demo we'll show structure
        logger.info("🦙 LLaMA-2-7B Architecture:")
        logger.info("  Total Layers: 32")
        logger.info("  Hidden Size: 4096")
        logger.info("  Vocabulary: 32,000 tokens")
        logger.info("  Parameters: ~7 billion")
        
        logger.info("\n🔄 Federated Split Learning:")
        logger.info("  Client Side (per client):")
        logger.info("    ├── Embedding Layer: 32K → 4096")
        logger.info("    ├── Transformer Layers 0-15")
        logger.info("    ├── Layer Normalization")
        logger.info("    └── Privacy: Noise + Quantization")
        
        logger.info("  Server Side:")
        logger.info("    ├── Transformer Layers 16-31")
        logger.info("    ├── Final Normalization")
        logger.info("    └── LM Head: 4096 → 32K")
        
        logger.info("\n🔒 Privacy Mechanisms:")
        logger.info(f"  Gaussian Noise: σ = {Config.PRIVACY_SIGMA}")
        logger.info(f"  Quantization: {Config.QUANTIZATION_BITS} bits")
        logger.info("  Differential Privacy: (ε=10.0, δ=1e-5)")
        
        logger.info("\n📊 Non-IID Distribution:")
        logger.info(f"  Type: {Config.NON_IID_TYPE}")
        logger.info(f"  Concentration: α = {Config.NON_IID_ALPHA}")
        logger.info(f"  Clients: {Config.NUM_CLIENTS}")
        
    except Exception as e:
        logger.error(f"❌ Architecture demo failed: {e}")

def show_sample_data():
    """Show sample medical QA data"""
    logger = setup_logging()
    
    logger.info("\n📋 Sample Medical QA Data")
    logger.info("=" * 50)
    
    # Sample data for each dataset type
    samples = {
        'pubmed_qa': {
            'question': 'Does vitamin D supplementation reduce the risk of respiratory tract infections?',
            'answer': 'Yes, vitamin D supplementation reduces the risk of acute respiratory tract infections, particularly in individuals with vitamin D deficiency.',
            'source': 'PubMedQA'
        },
        'medqa': {
            'question': 'A 45-year-old man presents with chest pain and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. What is the most likely diagnosis?\nA) Anterior MI\nB) Inferior MI\nC) Lateral MI\nD) Posterior MI',
            'answer': 'B) Inferior MI - ST-elevation in leads II, III, and aVF indicates inferior wall myocardial infarction.',
            'source': 'MedQA'
        },
        'medmcqa': {
            'question': 'Which of the following is the first-line treatment for hypertension in a diabetic patient?\nA) ACE inhibitors\nB) Beta blockers\nC) Calcium channel blockers\nD) Thiazide diuretics',
            'answer': 'The correct answer is A: ACE inhibitors\n\nExplanation: ACE inhibitors are first-line for diabetic patients as they provide renal protection.',
            'source': 'MedMCQA'
        }
    }
    
    for dataset_name, sample in samples.items():
        logger.info(f"\n📖 {dataset_name.upper()} Sample:")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Answer: {sample['answer']}")
        logger.info(f"Source: {sample['source']}")

def main():
    """Run all demos"""
    logger = setup_logging()
    
    logger.info("🎬 Starting Federated LLaMA-2 Medical QA Demo")
    
    # Show sample data
    show_sample_data()
    
    # Show model architecture
    demo_model_architecture()
    
    # Demo Non-IID distribution (only if not memory constrained)
    try:
        demo_non_iid_distribution()
    except Exception as e:
        logger.warning(f"⚠️ Skipping distribution demo due to: {e}")
        logger.info("💡 This is normal if you don't have access to datasets yet")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Demo completed! Ready to run full training:")
    logger.info("   python main.py --quick_test")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
