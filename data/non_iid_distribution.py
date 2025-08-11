"""
Non-IID Data Distribution for Federated Learning
Each client gets the same dataset but with different distributions
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Any
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class NonIIDDistributor:
    """
    Creates Non-IID data distribution for federated clients
    Each client gets biased samples from the same dataset
    """
    
    def __init__(self, dataset: Dataset, num_clients: int, distribution_type: str = "specialty"):
        self.dataset = dataset
        self.num_clients = num_clients
        self.distribution_type = distribution_type
        
        logger.info(f"Creating Non-IID distribution: {distribution_type}")
        
    def create_non_iid_splits(self, alpha: float = 0.5) -> List[Subset]:
        """
        Create non-IID splits using Dirichlet distribution
        
        Args:
            alpha: Dirichlet concentration parameter (lower = more non-IID)
                  - alpha < 1.0: Highly non-IID
                  - alpha = 1.0: Moderately non-IID  
                  - alpha > 1.0: Closer to IID
        
        Returns:
            List of dataset subsets for each client
        """
        if self.distribution_type == "specialty":
            return self._create_specialty_based_splits(alpha)
        elif self.distribution_type == "difficulty":
            return self._create_difficulty_based_splits(alpha)
        elif self.distribution_type == "question_type":
            return self._create_question_type_splits(alpha)
        else:
            return self._create_dirichlet_splits(alpha)
    
    def _create_specialty_based_splits(self, alpha: float) -> List[Subset]:
        """Create splits based on medical specialties"""
        logger.info("Creating specialty-based Non-IID splits...")
        
        # Define medical specialties and keywords
        specialties = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'artery', 'blood pressure'],
            'neurology': ['brain', 'neural', 'nerve', 'neurological', 'seizure', 'stroke'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'oncology', 'carcinoma'],
            'infectious': ['infection', 'virus', 'bacteria', 'antibiotic', 'fever', 'pathogen'],
            'endocrine': ['diabetes', 'hormone', 'thyroid', 'insulin', 'glucose', 'endocrine'],
            'respiratory': ['lung', 'respiratory', 'breathing', 'pneumonia', 'asthma', 'oxygen'],
            'gastro': ['stomach', 'digestive', 'intestinal', 'liver', 'gastric', 'bowel'],
            'general': []  # Everything else
        }
        
        # Categorize samples by specialty
        specialty_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            question = sample['question'].lower()
            answer = sample['answer'].lower()
            text = f"{question} {answer}"
            
            # Find matching specialty
            matched_specialty = 'general'
            for specialty, keywords in specialties.items():
                if specialty == 'general':
                    continue
                if any(keyword in text for keyword in keywords):
                    matched_specialty = specialty
                    break
            
            specialty_indices[matched_specialty].append(idx)
        
        # Log specialty distribution
        for specialty, indices in specialty_indices.items():
            logger.info(f"  {specialty}: {len(indices)} samples")
        
        # Create non-IID distribution using Dirichlet
        client_indices = [[] for _ in range(self.num_clients)]
        
        for specialty, indices in specialty_indices.items():
            if len(indices) == 0:
                continue
                
            # Generate Dirichlet distribution for this specialty
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Distribute indices according to proportions
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + int(len(indices) * proportions[client_id])
                if client_id == self.num_clients - 1:  # Last client gets remainder
                    end_idx = len(indices)
                
                client_indices[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create subsets
        client_datasets = []
        for client_id in range(self.num_clients):
            # Shuffle client's indices
            np.random.shuffle(client_indices[client_id])
            subset = Subset(self.dataset, client_indices[client_id])
            client_datasets.append(subset)
            
            # Log client's specialty distribution
            client_specialties = defaultdict(int)
            for idx in client_indices[client_id]:
                sample = self.dataset[idx]
                question = sample['question'].lower()
                answer = sample['answer'].lower()
                text = f"{question} {answer}"
                
                matched_specialty = 'general'
                for specialty, keywords in specialties.items():
                    if specialty == 'general':
                        continue
                    if any(keyword in text for keyword in keywords):
                        matched_specialty = specialty
                        break
                client_specialties[matched_specialty] += 1
            
            logger.info(f"Client {client_id}: {len(client_indices[client_id])} samples")
            for specialty, count in client_specialties.items():
                if count > 0:
                    logger.info(f"  {specialty}: {count}")
        
        return client_datasets
    
    def _create_difficulty_based_splits(self, alpha: float) -> List[Subset]:
        """Create splits based on question difficulty"""
        logger.info("Creating difficulty-based Non-IID splits...")
        
        # Estimate difficulty based on text length and complexity
        difficulties = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            question = sample['question']
            answer = sample['answer']
            
            # Simple difficulty heuristics
            q_len = len(question.split())
            a_len = len(answer.split())
            
            # Complex medical terms indicator
            complex_terms = ['syndrome', 'pathophysiology', 'etiology', 'differential', 'prognosis']
            complexity_score = sum(1 for term in complex_terms if term in question.lower() or term in answer.lower())
            
            # Difficulty score (normalized)
            difficulty = (q_len + a_len + complexity_score * 10) / 100
            difficulties.append((idx, difficulty))
        
        # Sort by difficulty
        difficulties.sort(key=lambda x: x[1])
        
        # Create difficulty categories
        total_samples = len(difficulties)
        easy_samples = difficulties[:total_samples//3]
        medium_samples = difficulties[total_samples//3:2*total_samples//3]
        hard_samples = difficulties[2*total_samples//3:]
        
        difficulty_groups = {
            'easy': [idx for idx, _ in easy_samples],
            'medium': [idx for idx, _ in medium_samples],
            'hard': [idx for idx, _ in hard_samples]
        }
        
        # Log difficulty distribution
        for difficulty, indices in difficulty_groups.items():
            logger.info(f"  {difficulty}: {len(indices)} samples")
        
        # Distribute using Dirichlet (similar to specialty-based)
        client_indices = [[] for _ in range(self.num_clients)]
        
        for difficulty, indices in difficulty_groups.items():
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            np.random.shuffle(indices)
            
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + int(len(indices) * proportions[client_id])
                if client_id == self.num_clients - 1:
                    end_idx = len(indices)
                
                client_indices[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create subsets
        client_datasets = []
        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])
            subset = Subset(self.dataset, client_indices[client_id])
            client_datasets.append(subset)
            logger.info(f"Client {client_id}: {len(client_indices[client_id])} samples")
        
        return client_datasets
    
    def _create_question_type_splits(self, alpha: float) -> List[Subset]:
        """Create splits based on question types"""
        logger.info("Creating question-type-based Non-IID splits...")
        
        # Define question types based on patterns
        question_types = {
            'diagnosis': ['what is the diagnosis', 'diagnose', 'most likely diagnosis', 'differential diagnosis'],
            'treatment': ['treatment', 'therapy', 'medication', 'drug', 'prescribe'],
            'symptom': ['symptom', 'sign', 'presentation', 'manifest'],
            'mechanism': ['mechanism', 'pathophysiology', 'how does', 'why does'],
            'epidemiology': ['prevalence', 'incidence', 'risk factor', 'epidemiology'],
            'other': []
        }
        
        # Categorize questions
        type_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            question = sample['question'].lower()
            
            matched_type = 'other'
            for qtype, patterns in question_types.items():
                if qtype == 'other':
                    continue
                if any(pattern in question for pattern in patterns):
                    matched_type = qtype
                    break
            
            type_indices[matched_type].append(idx)
        
        # Log type distribution
        for qtype, indices in type_indices.items():
            logger.info(f"  {qtype}: {len(indices)} samples")
        
        # Distribute using Dirichlet
        client_indices = [[] for _ in range(self.num_clients)]
        
        for qtype, indices in type_indices.items():
            if len(indices) == 0:
                continue
                
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            np.random.shuffle(indices)
            
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + int(len(indices) * proportions[client_id])
                if client_id == self.num_clients - 1:
                    end_idx = len(indices)
                
                client_indices[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create subsets
        client_datasets = []
        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])
            subset = Subset(self.dataset, client_indices[client_id])
            client_datasets.append(subset)
            logger.info(f"Client {client_id}: {len(client_indices[client_id])} samples")
        
        return client_datasets
    
    def _create_dirichlet_splits(self, alpha: float) -> List[Subset]:
        """Create general Dirichlet-based splits"""
        logger.info("Creating general Dirichlet Non-IID splits...")
        
        # Simple random partitioning with Dirichlet distribution
        total_samples = len(self.dataset)
        proportions = np.random.dirichlet([alpha] * self.num_clients)
        
        # Create indices
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        client_datasets = []
        start_idx = 0
        
        for client_id in range(self.num_clients):
            end_idx = start_idx + int(total_samples * proportions[client_id])
            if client_id == self.num_clients - 1:  # Last client gets remainder
                end_idx = total_samples
            
            client_indices = indices[start_idx:end_idx]
            subset = Subset(self.dataset, client_indices)
            client_datasets.append(subset)
            
            logger.info(f"Client {client_id}: {len(client_indices)} samples")
            start_idx = end_idx
        
        return client_datasets
    
    def analyze_distribution(self, client_datasets: List[Subset]) -> Dict[str, Any]:
        """Analyze the distribution characteristics"""
        logger.info("Analyzing Non-IID distribution...")
        
        analysis = {
            'client_sizes': [len(ds) for ds in client_datasets],
            'size_std': np.std([len(ds) for ds in client_datasets]),
            'distribution_type': self.distribution_type
        }
        
        logger.info(f"Client sizes: {analysis['client_sizes']}")
        logger.info(f"Size standard deviation: {analysis['size_std']:.2f}")
        
        return analysis
