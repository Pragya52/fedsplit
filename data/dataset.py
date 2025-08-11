"""
Medical QA Dataset Handler - Single Dataset Focus
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class MedicalQADataset(Dataset):
    """
    Medical QA Dataset - Focused on single dataset at a time
    Currently supports: PubMedQA, MedQA, MedMCQA
    """
    
    def __init__(self, tokenizer, max_length: int = None, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or Config.MAX_LENGTH
        self.max_samples = max_samples or Config.MAX_SAMPLES
        self.data = []
        
        # Get dataset configuration
        self.dataset_config = Config.get_dataset_config()
        
        logger.info(f"Loading {Config.DATASET_NAME} dataset...")
        self._load_dataset()
        logger.info(f"✅ Loaded {len(self.data)} samples from {Config.DATASET_NAME}")
    
    def _load_dataset(self):
        """Load the selected dataset"""
        try:
            # Load dataset from HuggingFace
            if self.dataset_config["subset"]:
                dataset = load_dataset(
                    self.dataset_config["dataset_path"],
                    self.dataset_config["subset"],
                    split=self.dataset_config["split"]
                )
            else:
                dataset = load_dataset(
                    self.dataset_config["dataset_path"],
                    split=self.dataset_config["split"]
                )
            
            # Process based on dataset type
            if Config.DATASET_NAME == "pubmed_qa":
                self._process_pubmed_qa(dataset)
            elif Config.DATASET_NAME == "medqa":
                self._process_medqa(dataset)
            elif Config.DATASET_NAME == "medmcqa":
                self._process_medmcqa(dataset)
            
        except Exception as e:
            logger.error(f"Failed to load {Config.DATASET_NAME}: {e}")
            raise
    
    def _process_pubmed_qa(self, dataset):
        """Process PubMedQA dataset"""
        count = 0
        for item in dataset:
            if count >= self.max_samples:
                break
            
            # Skip uncertain answers
            if item.get('final_decision') == 'maybe':
                continue
            
            question = item[self.dataset_config["question_key"]]
            
            # Use long answer if available, otherwise final decision
            answer = item.get(self.dataset_config["answer_key"])
            if not answer or answer.strip() == "":
                answer = f"The answer is {item['final_decision']}."
            
            # Add context if available
            context = ""
            if "context" in item and item["context"]:
                if isinstance(item["context"], dict) and "contexts" in item["context"]:
                    context = " ".join(item["context"]["contexts"][:2])  # Limit context
                elif isinstance(item["context"], list):
                    context = " ".join(item["context"][:2])
            
            if context:
                question = f"Context: {context}\n\nQuestion: {question}"
            
            self.data.append({
                'question': question,
                'answer': answer,
                'source': 'PubMedQA'
            })
            count += 1
    
    def _process_medqa(self, dataset):
        """Process MedQA dataset"""
        count = 0
        for item in dataset:
            if count >= self.max_samples:
                break
            
            question = item[self.dataset_config["question_key"]]
            answer = item[self.dataset_config["answer_key"]]
            
            # Add options if available
            if "options" in item and item["options"]:
                options_text = "\n".join([f"{k}: {v}" for k, v in item["options"].items()])
                question = f"{question}\n\nOptions:\n{options_text}"
            
            self.data.append({
                'question': question,
                'answer': answer,
                'source': 'MedQA'
            })
            count += 1
    
    def _process_medmcqa(self, dataset):
        """Process MedMCQA dataset"""
        count = 0
        for item in dataset:
            if count >= self.max_samples:
                break
            
            question = item[self.dataset_config["question_key"]]
            correct_idx = item[self.dataset_config["answer_key"]]
            
            # Get options
            options = [
                item.get('opa', ''),
                item.get('opb', ''),
                item.get('opc', ''),
                item.get('opd', '')
            ]
            
            # Filter out empty options
            valid_options = [(i, opt) for i, opt in enumerate(options) if opt.strip()]
            
            if len(valid_options) < 2 or correct_idx >= len(options):
                continue
            
            correct_answer = options[correct_idx]
            
            # Format question with options
            options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options) if opt.strip()])
            formatted_question = f"{question}\n\n{options_text}"
            
            # Create answer
            explanation = item.get('exp', '')
            if explanation and explanation.strip():
                answer = f"The correct answer is {chr(65+correct_idx)}: {correct_answer}\n\nExplanation: {explanation}"
            else:
                answer = f"The correct answer is {chr(65+correct_idx)}: {correct_answer}"
            
            self.data.append({
                'question': formatted_question,
                'answer': answer,
                'source': 'MedMCQA'
            })
            count += 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        answer = item['answer']
        
        # Create training format
        full_text = f"Question: {question}\nAnswer: {answer}"
        prompt_text = f"Question: {question}\nAnswer:"
        
        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize prompt only (for computing labels)
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (mask prompt part)
        labels = encoding['input_ids'].clone()
        prompt_length = len(prompt_encoding['input_ids'][0])
        
        # Only predict answer tokens
        if prompt_length < len(labels[0]):
            labels[0, :prompt_length] = -100
        else:
            # If prompt is too long, mask everything
            labels[0, :] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'labels': labels.squeeze(0),
            'question': question,
            'answer': answer,
            'source': item['source']
        }
