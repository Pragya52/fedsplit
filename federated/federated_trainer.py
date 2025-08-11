"""
Federated Learning Trainer for LLaMA-2 Medical QA
Fixed tensor mismatches and improved architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import logging
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List, Any

from config.config import Config
from data.dataset import MedicalQADataset
from data.non_iid_distribution import NonIIDDistributor
from models.llama_models import LLaMAModelManager
from federated.privacy import PrivacyMechanism, DifferentialPrivacyAccountant
from utils.metrics import MedicalQAMetrics

logger = logging.getLogger(__name__)

class FederatedTrainer:
    """
    Main Federated Learning Trainer
    """
    
    def __init__(self):
        self.device = Config.DEVICE
        Config.setup_directories()
        
        logger.info("🚀 Initializing Federated LLaMA-2 Medical QA Trainer")
        
        # Initialize components
        self._setup_model_manager()
        self._setup_data()
        self._setup_models()
        self._setup_privacy()
        self._setup_training()
        
        # Training state
        self.current_round = 0
        self.best_accuracy = 0.0
        self.training_history = []
    
    def _setup_model_manager(self):
        """Setup model manager"""
        self.model_manager = LLaMAModelManager()
        self.tokenizer = self.model_manager.tokenizer
        
    def _setup_data(self):
        """Setup federated data distribution with Non-IID"""
        logger.info(f"Setting up Non-IID data for {Config.DATASET_NAME}...")
        
        # Create full dataset
        full_dataset = MedicalQADataset(
            tokenizer=self.tokenizer,
            max_length=Config.MAX_LENGTH,
            max_samples=Config.MAX_SAMPLES
        )
        
        # Train/test split
        total_size = len(full_dataset)
        test_size = int(Config.TEST_RATIO * total_size)
        train_size = total_size - test_size
        
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Create Non-IID distribution for clients
        logger.info(f"Creating Non-IID distribution: {Config.NON_IID_TYPE} (α={Config.NON_IID_ALPHA})")
        
        non_iid_distributor = NonIIDDistributor(
            dataset=train_dataset,
            num_clients=Config.NUM_CLIENTS,
            distribution_type=Config.NON_IID_TYPE
        )
        
        # Get Non-IID client datasets
        client_datasets = non_iid_distributor.create_non_iid_splits(
            alpha=Config.NON_IID_ALPHA
        )
        
        # Analyze distribution
        distribution_analysis = non_iid_distributor.analyze_distribution(client_datasets)
        
        # Create data loaders
        self.client_loaders = []
        for client_id, dataset in enumerate(client_datasets):
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            self.client_loaders.append(loader)
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Log distribution info
        client_sizes = [len(ds) for ds in client_datasets]
        logger.info(f"✅ Non-IID Data Distribution:")
        logger.info(f"  Distribution Type: {Config.NON_IID_TYPE}")
        logger.info(f"  Alpha (concentration): {Config.NON_IID_ALPHA}")
        logger.info(f"  Client sizes: {client_sizes}")
        logger.info(f"  Size std deviation: {distribution_analysis['size_std']:.2f}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        
        # Store distribution info for results
        self.distribution_info = {
            'type': Config.NON_IID_TYPE,
            'alpha': Config.NON_IID_ALPHA,
            'client_sizes': client_sizes,
            'analysis': distribution_analysis
        }
    
    def _setup_models(self):
        """Setup federated models"""
        logger.info("Setting up federated models...")
        
        # Create client models
        self.clients = []
        for i in range(Config.NUM_CLIENTS):
            client = self.model_manager.create_client_model()
            self.clients.append(client)
        
        # Create server model
        self.server = self.model_manager.create_server_model()
        
        logger.info(f"✅ Created {Config.NUM_CLIENTS} clients + 1 server")
    
    def _setup_privacy(self):
        """Setup privacy mechanisms"""
        self.privacy_mechanism = PrivacyMechanism()
        self.dp_accountant = DifferentialPrivacyAccountant()
        
    def _setup_training(self):
        """Setup optimizers and metrics"""
        # Client optimizers
        self.client_optimizers = []
        for client in self.clients:
            optimizer = optim.AdamW(
                client.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
            self.client_optimizers.append(optimizer)
        
        # Server optimizer
        self.server_optimizer = optim.AdamW(
            self.server.parameters(),
            lr=Config.LEARNING_RATE * 0.5,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = MedicalQAMetrics(self.tokenizer)
        
        logger.info(f"✅ Training setup complete")
    
    def train_client_locally(self, client_id: int, epochs: int = None) -> float:
        """Train a single client locally"""
        epochs = epochs or Config.LOCAL_EPOCHS
        
        client = self.clients[client_id]
        optimizer = self.client_optimizers[client_id]
        dataloader = self.client_loaders[client_id]
        
        client.train()
        self.server.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Client {client_id} Epoch {epoch+1}", leave=False):
                try:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass through client
                    client_hidden = client(input_ids, attention_mask)
                    
                    # Apply privacy mechanisms
                    private_hidden = self.privacy_mechanism.apply_privacy(client_hidden)
                    
                    # Forward pass through server
                    server_logits = self.server(private_hidden, attention_mask)
                    
                    # Compute loss
                    loss = self.loss_fn(
                        server_logits.view(-1, server_logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    self.server_optimizer.zero_grad()
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(client.parameters(), Config.GRADIENT_CLIP)
                    torch.nn.utils.clip_grad_norm_(self.server.parameters(), Config.GRADIENT_CLIP)
                    
                    optimizer.step()
                    self.server_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM in client {client_id}, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def federated_averaging(self):
        """Perform federated averaging of client parameters"""
        if len(self.clients) <= 1:
            return
        
        logger.info("Performing federated averaging...")
        
        # Collect parameters from all clients
        client_params_list = []
        for client in self.clients:
            params = client.get_aggregatable_parameters()
            client_params_list.append(params)
        
        # Average parameters
        averaged_params = {}
        for key in client_params_list[0].keys():
            param_tensors = [params[key] for params in client_params_list if params[key] is not None]
            if param_tensors:
                averaged_params[key] = torch.stack(param_tensors).mean(dim=0)
        
        # Distribute averaged parameters back to clients
        for client in self.clients:
            client.set_aggregated_parameters(averaged_params)
        
        logger.info("✅ Federated averaging completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set"""
        client = self.clients[0]  # Use first client as representative
        client.eval()
        self.server.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass (no privacy during evaluation)
                    client_hidden = client(input_ids, attention_mask)
                    server_logits = self.server(client_hidden, attention_mask)
                    
                    # Compute metrics
                    loss = self.loss_fn(
                        server_logits.view(-1, server_logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    accuracy = self.metrics.compute_accuracy(server_logits, labels)
                    perplexity = self.metrics.compute_perplexity(server_logits, labels)
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    total_perplexity += perplexity
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM during evaluation, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if num_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
        
        results = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'perplexity': total_perplexity / num_batches
        }
        
        return results
    
    def save_model(self, path: Path, is_best: bool = False):
        """Save model checkpoints"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save client and server models
        torch.save(self.clients[0].state_dict(), path / "client_model.pt")
        torch.save(self.server.state_dict(), path / "server_model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save training info
        training_info = {
            'round': self.current_round,
            'best_accuracy': self.best_accuracy,
            'config': {
                'model_name': Config.MODEL_NAME,
                'dataset_name': Config.DATASET_NAME,
                'split_layer': Config.SPLIT_LAYER,
                'num_clients': Config.NUM_CLIENTS
            }
        }
        
        with open(path / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        if is_best:
            logger.info(f"💾 Best model saved to {path}")
        else:
            logger.info(f"📁 Checkpoint saved to {path}")
    
    def train_federated(self):
        """Main federated training loop"""
        logger.info(f"🚀 Starting federated training for {Config.NUM_ROUNDS} rounds")
        
        training_start = time.time()
        
        for round_num in range(1, Config.NUM_ROUNDS + 1):
            round_start = time.time()
            self.current_round = round_num
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ROUND {round_num}/{Config.NUM_ROUNDS}")
            logger.info(f"{'='*60}")
            
            # Check privacy budget
            if not self.dp_accountant.account_privacy_round(Config.PRIVACY_SIGMA):
                logger.warning("⚠️ Privacy budget exhausted, stopping training")
                break
            
            # Train each client locally
            round_losses = []
            for client_id in range(Config.NUM_CLIENTS):
                logger.info(f"Training Client {client_id}...")
                
                client_loss = self.train_client_locally(client_id)
                round_losses.append(client_loss)
                
                logger.info(f"  Client {client_id} Loss: {client_loss:.4f}")
            
            # Federated averaging
            self.federated_averaging()
            
            # Evaluation
            if round_num % Config.EVAL_FREQUENCY == 0:
                eval_results = self.evaluate()
                
                logger.info(f"📊 Round {round_num} Evaluation:")
                logger.info(f"  Loss: {eval_results['loss']:.4f}")
                logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                
                # Save training history
                self.training_history.append({
                    'round': round_num,
                    'client_losses': round_losses,
                    'eval_results': eval_results,
                    'privacy_spent': self.dp_accountant.privacy_spent
                })
                
                # Save best model
                if eval_results['accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['accuracy']
                    self.save_model(Config.RESULTS_DIR / "best_model", is_best=True)
                    logger.info(f"💾 New best model! Accuracy: {self.best_accuracy:.4f}")
            
            # Save checkpoint
            if round_num % 5 == 0:
                self.save_model(Config.RESULTS_DIR / f"checkpoint_round_{round_num}")
            
            round_time = time.time() - round_start
            logger.info(f"⏱️ Round {round_num} completed in {round_time:.1f}s")
        
        total_time = time.time() - training_start
        
        # Save final results
        final_results = {
            'total_time_hours': total_time / 3600,
            'best_accuracy': self.best_accuracy,
            'rounds_completed': self.current_round,
            'privacy_status': self.dp_accountant.get_privacy_status(),
            'training_history': self.training_history
        }
        
        with open(Config.RESULTS_DIR / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\n🎉 TRAINING COMPLETED!")
        logger.info(f"  Total time: {total_time/3600:.2f} hours")
        logger.info(f"  Best accuracy: {self.best_accuracy:.4f}")
        logger.info(f"  Rounds completed: {self.current_round}")
        logger.info(f"  Privacy spent: ε={self.dp_accountant.privacy_spent:.3f}")
        
        return final_results
