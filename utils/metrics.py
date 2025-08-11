"""
Evaluation Metrics for Medical QA
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class MedicalQAMetrics:
    """
    Evaluation metrics for medical question answering
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute token-level accuracy
        
        Args:
            predictions: Model logits [batch_size, seq_len, vocab_size]
            labels: True labels [batch_size, seq_len]
        
        Returns:
            Token accuracy as float
        """
        # Get predicted tokens
        pred_tokens = torch.argmax(predictions, dim=-1)
        
        # Create mask for valid tokens (ignore -100)
        mask = (labels != -100)
        
        if mask.sum() == 0:
            return 0.0
        
        # Calculate accuracy only on valid tokens
        correct = (pred_tokens[mask] == labels[mask]).float()
        accuracy = correct.mean().item()
        
        return accuracy
    
    def compute_perplexity(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute perplexity
        
        Args:
            predictions: Model logits [batch_size, seq_len, vocab_size]
            labels: True labels [batch_size, seq_len]
        
        Returns:
            Perplexity as float
        """
        # Compute cross entropy loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            predictions.view(-1, predictions.size(-1)),
            labels.view(-1)
        )
        
        # Convert to perplexity
        perplexity = torch.exp(loss).item()
        
        # Cap perplexity to avoid overflow
        return min(perplexity, 10000.0)
    
    def compute_bleu_score(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute BLEU score for generated answers
        Simple implementation for medical QA evaluation
        """
        try:
            # Get predicted tokens
            pred_tokens = torch.argmax(predictions, dim=-1)
            
            # Convert to text
            batch_bleu_scores = []
            
            for i in range(pred_tokens.size(0)):
                # Get valid prediction tokens
                pred_seq = pred_tokens[i]
                label_seq = labels[i]
                
                # Remove padding and special tokens
                pred_valid = pred_seq[pred_seq != self.tokenizer.pad_token_id]
                label_valid = label_seq[label_seq != -100]
                
                if len(pred_valid) == 0 or len(label_valid) == 0:
                    batch_bleu_scores.append(0.0)
                    continue
                
                # Convert to text
                pred_text = self.tokenizer.decode(pred_valid, skip_special_tokens=True)
                label_text = self.tokenizer.decode(label_valid, skip_special_tokens=True)
                
                # Simple word-level BLEU-1
                pred_words = set(pred_text.lower().split())
                label_words = set(label_text.lower().split())
                
                if len(label_words) == 0:
                    bleu_score = 0.0
                else:
                    intersection = pred_words.intersection(label_words)
                    bleu_score = len(intersection) / len(label_words)
                
                batch_bleu_scores.append(bleu_score)
            
            return np.mean(batch_bleu_scores)
            
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return 0.0
    
    def compute_medical_keywords_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute accuracy on medical keywords/terms
        """
        # Common medical keywords to check
        medical_keywords = {
            'treatment', 'diagnosis', 'symptoms', 'patient', 'disease',
            'therapy', 'medication', 'clinical', 'medical', 'health',
            'condition', 'syndrome', 'disorder', 'infection', 'virus',
            'bacteria', 'cancer', 'tumor', 'surgery', 'procedure'
        }
        
        try:
            pred_tokens = torch.argmax(predictions, dim=-1)
            
            keyword_scores = []
            
            for i in range(pred_tokens.size(0)):
                pred_seq = pred_tokens[i]
                label_seq = labels[i]
                
                # Convert to text
                pred_valid = pred_seq[pred_seq != self.tokenizer.pad_token_id]
                label_valid = label_seq[label_seq != -100]
                
                if len(pred_valid) == 0 or len(label_valid) == 0:
                    keyword_scores.append(0.0)
                    continue
                
                pred_text = self.tokenizer.decode(pred_valid, skip_special_tokens=True).lower()
                label_text = self.tokenizer.decode(label_valid, skip_special_tokens=True).lower()
                
                # Find medical keywords in both texts
                pred_medical_words = {word for word in pred_text.split() if word in medical_keywords}
                label_medical_words = {word for word in label_text.split() if word in medical_keywords}
                
                if len(label_medical_words) == 0:
                    keyword_scores.append(1.0 if len(pred_medical_words) == 0 else 0.0)
                else:
                    intersection = pred_medical_words.intersection(label_medical_words)
                    score = len(intersection) / len(label_medical_words)
                    keyword_scores.append(score)
            
            return np.mean(keyword_scores)
            
        except Exception as e:
            logger.warning(f"Medical keyword accuracy computation failed: {e}")
            return 0.0
    
    def compute_all_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics at once
        
        Returns:
            Dictionary with all computed metrics
        """
        return {
            'accuracy': self.compute_accuracy(predictions, labels),
            'perplexity': self.compute_perplexity(predictions, labels),
            'bleu_score': self.compute_bleu_score(predictions, labels),
            'medical_keywords_accuracy': self.compute_medical_keywords_accuracy(predictions, labels)
        }
