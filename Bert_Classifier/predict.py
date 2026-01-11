# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
import os
import logging
import time

# --- Logging and basic settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# This configuration must exactly match the training setup
CONFIG = {
    "model_name": "xlm-roberta-base",
    "max_total_length": 1024,
    "chunk_size": 512,
    "response_s_l_boundary": 444.0
}

# --- 2. Model definition ---
# The model architecture must be identical to the one used during training
class HierarchicalModel(nn.Module):
    def __init__(self, encoder_name, num_labels=1):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(encoder_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        self.aggregator = nn.LSTM(
            input_size=encoder_hidden_size, hidden_size=encoder_hidden_size // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.regressor_head = nn.Linear(encoder_hidden_size, num_labels)

    def forward(self, input_ids_chunks, attention_mask_chunks, valid_chunks):
        B, T, L = input_ids_chunks.shape
        input_ids_flat, attention_mask_flat = input_ids_chunks.view(-1, L), attention_mask_chunks.view(-1, L)
        chunk_repr = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat).last_hidden_state[:, 0, :]
        seq_repr = chunk_repr.view(B, T, -1)
        lengths = valid_chunks.clamp(min=1).cpu()
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_repr[idx_sort], lengths_sorted.tolist(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, c_n) = self.aggregator(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        _, idx_unsort = torch.sort(idx_sort)
        return self.regressor_head(h_last[idx_unsort])

def _preprocess_prompt(prompt_text, tokenizer, config):
    """
    (Internal) Convert a single prompt string into the model input format (tensors).
    """
    max_tokens_per_chunk = config['chunk_size'] - 2
    num_chunks = config['max_total_length'] // max_tokens_per_chunk
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False, truncation=True, max_length=max_tokens_per_chunk * num_chunks)
    chunks_of_ids = [token_ids[i: i + max_tokens_per_chunk] for i in range(0, len(token_ids), max_tokens_per_chunk)][:num_chunks]
    valid_chunks_count = len(chunks_of_ids)

    all_input_ids, all_attention_masks = [], []
    for chunk in chunks_of_ids:
        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        padding_len = config['chunk_size'] - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_len
        attention_mask += [0] * padding_len
        all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        all_attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

    for _ in range(num_chunks - len(all_input_ids)):
        all_input_ids.append(torch.full((config['chunk_size'],), tokenizer.pad_token_id, dtype=torch.long))
        all_attention_masks.append(torch.zeros(config['chunk_size'], dtype=torch.long))

    return {
        'input_ids_chunks': torch.stack(all_input_ids),
        'attention_mask_chunks': torch.stack(all_attention_masks),
        'valid_chunks': torch.tensor(valid_chunks_count, dtype=torch.long)
    }

class TokenCountPredictor:
    """
    A predictor that wraps a hierarchical Transformer regression model.
    """
    def __init__(self, model_dir: str):
        """
        Initialize the predictor, loading the model and tokenizer.
        :param model_dir: Directory that contains 'best_model.pt' and the tokenizer files.
        """
        logger.info(f"Initializing TokenCountPredictor...")
        model_path = os.path.join(model_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: model checkpoint file 'best_model.pt' not found in '{model_dir}'.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = CONFIG
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading tokenizer from '{model_dir}'...")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)

        logger.info(f"Initializing model '{self.config['model_name']}'...")
        self.model = HierarchicalModel(encoder_name=self.config['model_name'], num_labels=1)
        
        logger.info(f"Loading model weights from '{model_path}'...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("TokenCountPredictor initialization complete.")

    def predict(self, prompt_text: str) -> int:
        """
        Predict the number of response tokens for the given prompt text.
        :param prompt_text: Input prompt string.
        :return: Predicted token count (int).
        """
        inputs = _preprocess_prompt(prompt_text, self.tokenizer, self.config)
        
        input_ids_chunks = inputs['input_ids_chunks'].unsqueeze(0).to(self.device)
        attention_mask_chunks = inputs['attention_mask_chunks'].unsqueeze(0).to(self.device)
        valid_chunks = inputs['valid_chunks'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction_log = self.model(
                input_ids_chunks=input_ids_chunks,
                attention_mask_chunks=attention_mask_chunks,
                valid_chunks=valid_chunks
            )
        
        predicted_log_value = prediction_log.squeeze().cpu().numpy()
        predicted_token_count = np.expm1(predicted_log_value)
        
        return int(round(predicted_token_count))
    
    def benchmark(self, prompt_text: str, warmup: int = 2, repeat: int = 10):
        """
        Timing evaluation:
        - forward_only_ms: forward pass only (excluding tokenization / device transfer)
        - end_to_end_ms: end-to-end (including tokenization, input packing, device transfer, forward pass, and CPU transfer)
        Returns mean and std in milliseconds.
        """
        # ---------- Prepare inputs once for forward-only timing ----------
        inputs = _preprocess_prompt(prompt_text, self.tokenizer, self.config)
        input_ids_chunks = inputs['input_ids_chunks'].unsqueeze(0).to(self.device, non_blocking=True)
        attention_mask_chunks = inputs['attention_mask_chunks'].unsqueeze(0).to(self.device, non_blocking=True)
        valid_chunks = inputs['valid_chunks'].unsqueeze(0).to(self.device, non_blocking=True)

        # Warm-up (avoid first-run allocation / kernel compilation noise)
        for _ in range(max(0, warmup)):
            with torch.no_grad():
                _ = self.model(
                    input_ids_chunks=input_ids_chunks,
                    attention_mask_chunks=attention_mask_chunks,
                    valid_chunks=valid_chunks
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # ---------- Forward-only timing ----------
        forward_times_ms = []
        for _ in range(repeat):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(
                        input_ids_chunks=input_ids_chunks,
                        attention_mask_chunks=attention_mask_chunks,
                        valid_chunks=valid_chunks
                    )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(
                        input_ids_chunks=input_ids_chunks,
                        attention_mask_chunks=attention_mask_chunks,
                        valid_chunks=valid_chunks
                    )
                t1 = time.perf_counter()
            forward_times_ms.append((t1 - t0) * 1000.0)

        forward_mean = float(np.mean(forward_times_ms))
        forward_std = float(np.std(forward_times_ms))

        # ---------- End-to-end timing (tokenize/pack/transfer/forward) ----------
        e2e_times_ms = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = self.predict(prompt_text)  # This call includes tokenization / packing / device transfer / forward pass
            t1 = time.perf_counter()
            e2e_times_ms.append((t1 - t0) * 1000.0)

        e2e_mean = float(np.mean(e2e_times_ms))
        e2e_std = float(np.std(e2e_times_ms))

        return {
            "forward_only_ms_mean": forward_mean,
            "forward_only_ms_std": forward_std,
            "end_to_end_ms_mean": e2e_mean,
            "end_to_end_ms_std": e2e_std,
            "repeat": repeat,
            "warmup": warmup,
        }

# --- Main entry point (standalone test) ---
if __name__ == '__main__':
    MODEL_OUTPUT_DIR = "bert_llama3_70B" # Name of your model directory
    
    if not os.path.isdir(MODEL_OUTPUT_DIR):
        logger.error(f"Error: model directory not found '{MODEL_OUTPUT_DIR}'.")
        logger.error(f"Please ensure this script and the directory '{MODEL_OUTPUT_DIR}' are in the same parent directory.")
        exit()

    # 1. Initialize predictor
    predictor = TokenCountPredictor(model_dir=MODEL_OUTPUT_DIR)

    # 2. Example prediction
    example_prompt = "Please explain what artificial intelligence is, its major subfields, and provide representative real-world applications in modern society. Be as detailed and comprehensive as possible."
    logger.info("\n" + "="*50)
    logger.info(f"Input prompt: \"{example_prompt[:80]}...\"")
    
    predicted_count = predictor.predict(example_prompt)
    category = "long" if predicted_count >= CONFIG['response_s_l_boundary'] else "short"

    logger.info(f"Predicted token count: {predicted_count}")
    logger.info(f"Predicted response type: [{category}]")
    logger.info("="*50 + "\n")

    bench = predictor.benchmark(example_prompt, warmup=2, repeat=10)
    logger.info("[Benchmark] repeat=%d, warmup=%d", bench["repeat"], bench["warmup"])
    logger.info("[Benchmark] Forward-only:  %.2f ± %.2f ms",
                bench["forward_only_ms_mean"], bench["forward_only_ms_std"])
    logger.info("[Benchmark] End-to-end:    %.2f ± %.2f ms",
                bench["end_to_end_ms_mean"], bench["end_to_end_ms_std"])
