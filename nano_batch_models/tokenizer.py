"""
Tokenizer wrapper for Mistral models.

This module provides a clean interface to HuggingFace tokenizers for encoding
and decoding text.
"""

from typing import List, Union, Optional
import torch


class MistralTokenizer:
    """
    Wrapper around HuggingFace AutoTokenizer for Mistral models.
    
    Provides convenient methods for encoding/decoding with proper handling
    of special tokens.
    """
    
    def __init__(self, model_name_or_path: str):
        """
        Initialize tokenizer.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
        """
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
            
        except ImportError:
            raise ImportError(
                "transformers package is required. "
                "Install with: pip install transformers"
            )
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: "pt" for PyTorch tensors, None for list
            
        Returns:
            Token IDs as list or tensor
        """
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return encoded
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ):
        """
        Encode multiple texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad sequences
            return_tensors: "pt" for PyTorch tensors, None for list
            
        Returns:
            Batch of encoded sequences
        """
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
        )
    
    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences.
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
