import gc
import logging
from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceRunner:
    """
    Encapsulates dynamic Hugging Face model and tokenizer lifecycle management,
    device placement (GPU/CUDA with automatic CPU fallback), and generation execution.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        """
        Initializes the model runner, detecting device placement and loading weights.

        Args:
            model_id: Hugging Face model identifier (e.g. 'gpt2', 'distilgpt2')
            device: Optional explicit device ('cuda', 'cpu'). Defaults to auto-detect.
        """
        self.model_id = model_id

        # Determine target device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model '{self.model_id}' onto device: {self.device}...")
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
        self.model: PreTrainedModel = self._load_model()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Loads and configures the Hugging Face tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Ensure pad_token is set (e.g., GPT-2 does not set pad_token by default)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """Loads and prepares the model for causal language modeling."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        )
        model.to(self.device)
        model.eval()
        return model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[str, int]:
        """
        Generates text given a prompt string and returns the decoded output text
        and generated token count.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of newly generated tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling (vs greedy decoding)

        Returns:
            Tuple[str, int]: (generated_text, generated_token_count)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs
            )

        # Slice out the prompt tokens so we measure purely newly generated output
        generated_token_ids = output_tokens[0][input_token_count:]
        generated_count = len(generated_token_ids)
        decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        return decoded_text, generated_count

    def unload(self) -> None:
        """
        Explicitly tears down model and tokenizer and reclaims GPU and RAM memory.
        Prevents cumulative memory accumulation and OOM across sequential model runs.
        """
        logger.info(f"Unloading model '{self.model_id}' and releasing memory...")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
