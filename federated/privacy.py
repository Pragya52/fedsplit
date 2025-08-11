"""
Privacy Mechanisms for Federated Learning
"""

import torch
import numpy as np
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class PrivacyMechanism:
    """
    Privacy mechanisms for federated learning
    - Gaussian noise for differential privacy
    - Quantization for communication efficiency
    """
    
    def __init__(self, sigma: float = None, quantization_bits: int = None):
        self.sigma = sigma or Config.PRIVACY_SIGMA
        self.quantization_bits = quantization_bits or Config.QUANTIZATION_BITS
        
        logger.info(f"✅ Privacy: σ={self.sigma}, {self.quantization_bits}-bit quantization")
    
    def add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for differential privacy"""
        if self.sigma <= 0:
            return tensor
        
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise
    
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to reduce communication overhead"""
        if self.quantization_bits >= 32:
            return tensor
        
        # Get tensor range
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max <= tensor_min:
            return tensor
        
        # Quantization levels
        max_val = 2**(self.quantization_bits - 1) - 1
        min_val = -2**(self.quantization_bits - 1)
        
        # Scale to quantization range
        scaled = (tensor - tensor_min) / (tensor_max - tensor_min)
        quantized = torch.round(scaled * (max_val - min_val) + min_val)
        
        # Reconstruct
        reconstructed = (quantized - min_val) / (max_val - min_val) * (tensor_max - tensor_min) + tensor_min
        
        return reconstructed
    
    def apply_privacy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply both noise and quantization"""
        # Add noise first
        noisy_tensor = self.add_gaussian_noise(tensor)
        
        # Then quantize
        private_tensor = self.quantize_tensor(noisy_tensor)
        
        return private_tensor

class DifferentialPrivacyAccountant:
    """
    Simple DP accountant for tracking privacy budget
    """
    
    def __init__(self, target_epsilon: float = 10.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.privacy_spent = 0.0
        self.rounds_completed = 0
    
    def account_privacy_round(self, sigma: float, sensitivity: float = 1.0):
        """Account for privacy spent in one round"""
        # Simple composition for Gaussian mechanism
        # ε = sensitivity^2 / (2 * σ^2)
        round_epsilon = (sensitivity ** 2) / (2 * sigma ** 2)
        
        self.privacy_spent += round_epsilon
        self.rounds_completed += 1
        
        logger.info(f"Privacy spent: ε={self.privacy_spent:.3f}/{self.target_epsilon}")
        
        return self.privacy_spent < self.target_epsilon
    
    def get_privacy_status(self):
        """Get current privacy status"""
        return {
            'epsilon_spent': self.privacy_spent,
            'epsilon_remaining': max(0, self.target_epsilon - self.privacy_spent),
            'rounds_completed': self.rounds_completed,
            'privacy_exhausted': self.privacy_spent >= self.target_epsilon
        }
