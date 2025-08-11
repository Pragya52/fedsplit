"""
LLaMA-2 Model Components for Federated Learning
Fixed tensor mismatch and memory issues
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class LLaMAModelManager:
    """
    Manages LLaMA-2 model loading and splitting
    Fixes memory issues by loading model once and sharing components
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.device = Config.DEVICE
        self.split_layer = Config.SPLIT_LAYER
        
        logger.info(f"🦙 Loading LLaMA-2-7B: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load full model once
        self._load_full_model()
        
        logger.info(f"✅ LLaMA-2 Manager initialized")
    
    def _load_full_model(self):
        """Load the full LLaMA-2 model once"""
        try:
            self.full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            self.config = self.full_model.config
            
            total_params = sum(p.numel() for p in self.full_model.parameters())
            logger.info(f"✅ Full model loaded: {total_params/1e6:.1f}M parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_client_model(self):
        """Create client model with first N layers"""
        return LLaMAClient(self.full_model, self.split_layer, self.device)
    
    def create_server_model(self):
        """Create server model with remaining layers"""
        return LLaMAServer(self.full_model, self.split_layer, self.device)

class LLaMAClient(nn.Module):
    """
    LLaMA-2 Client Model - First N layers
    Fixed tensor device issues
    """
    
    def __init__(self, full_model, split_layer: int, device):
        super().__init__()
        
        self.split_layer = split_layer
        self.device = device
        self.config = full_model.config
        
        # Copy embedding layer
        self.embed_tokens = nn.Embedding(
            full_model.model.embed_tokens.num_embeddings,
            full_model.model.embed_tokens.embedding_dim,
            padding_idx=full_model.model.embed_tokens.padding_idx
        )
        
        # Copy weights to new embedding
        with torch.no_grad():
            self.embed_tokens.weight.copy_(full_model.model.embed_tokens.weight)
        
        # Extract client layers (first split_layer layers)
        self.layers = nn.ModuleList()
        for i in range(min(split_layer, len(full_model.model.layers))):
            # Create new layer and copy weights
            original_layer = full_model.model.layers[i]
            new_layer = type(original_layer)(full_model.config, i)
            
            # Copy state dict
            new_layer.load_state_dict(original_layer.state_dict())
            self.layers.append(new_layer)
        
        # Add layer norm
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Move to device
        self.to(device)
        
        client_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ Client model: {client_params/1e6:.1f}M parameters")
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Expand attention mask for all heads
        batch_size, seq_len = input_ids.shape
        
        # Create 4D attention mask: [batch_size, 1, seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Combine with padding mask
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attention_mask_4d = attention_mask_4d.expand(batch_size, 1, seq_len, seq_len)
        
        # Apply causal mask
        attention_mask_4d = attention_mask_4d.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), False)
        
        # Convert to float and apply attention score mask
        attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
        attention_mask_4d = torch.where(
            attention_mask_4d, 
            torch.tensor(0.0, dtype=hidden_states.dtype, device=self.device),
            torch.tensor(-10000.0, dtype=hidden_states.dtype, device=self.device)
        )
        
        # Apply client layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            hidden_states = layer_outputs[0]
        
        # Apply normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def get_aggregatable_parameters(self):
        """Get parameters for federated averaging"""
        return {
            'embed_tokens.weight': self.embed_tokens.weight.clone().detach().cpu(),
            'norm.weight': self.norm.weight.clone().detach().cpu(),
        }
    
    def set_aggregated_parameters(self, params):
        """Set parameters from federated averaging"""
        with torch.no_grad():
            if 'embed_tokens.weight' in params:
                self.embed_tokens.weight.copy_(params['embed_tokens.weight'].to(self.device))
            if 'norm.weight' in params:
                self.norm.weight.copy_(params['norm.weight'].to(self.device))

class LLaMAServer(nn.Module):
    """
    LLaMA-2 Server Model - Remaining layers + head
    Fixed tensor device issues
    """
    
    def __init__(self, full_model, split_layer: int, device):
        super().__init__()
        
        self.split_layer = split_layer
        self.device = device
        self.config = full_model.config
        
        # Extract server layers (remaining layers)
        self.layers = nn.ModuleList()
        total_layers = len(full_model.model.layers)
        
        for i in range(split_layer, total_layers):
            # Create new layer and copy weights
            original_layer = full_model.model.layers[i]
            new_layer = type(original_layer)(full_model.config, i)
            
            # Copy state dict
            new_layer.load_state_dict(original_layer.state_dict())
            self.layers.append(new_layer)
        
        # Copy final norm and lm_head
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        with torch.no_grad():
            self.norm.weight.copy_(full_model.model.norm.weight)
        
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False
        )
        with torch.no_grad():
            self.lm_head.weight.copy_(full_model.lm_head.weight)
        
        # Move to device
        self.to(device)
        
        server_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ Server model: {server_params/1e6:.1f}M parameters")
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Ensure inputs are on correct device
        hidden_states = hidden_states.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
            # Create proper attention mask for server layers
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=1
            )
            
            # Expand attention mask
            attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            attention_mask_4d = attention_mask_4d.expand(batch_size, 1, seq_len, seq_len)
            
            # Apply causal mask
            attention_mask_4d = attention_mask_4d.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), False)
            
            # Convert to attention scores
            attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
            attention_mask_4d = torch.where(
                attention_mask_4d,
                torch.tensor(0.0, dtype=hidden_states.dtype, device=self.device),
                torch.tensor(-10000.0, dtype=hidden_states.dtype, device=self.device)
            )
        
        # Apply server layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d if attention_mask is not None else None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            hidden_states = layer_outputs[0]
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
