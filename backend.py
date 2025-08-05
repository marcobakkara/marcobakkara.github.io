import json
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFKC, Lowercase, Strip
from tokenizers.processors import TemplateProcessing
import logging
from typing import List, Dict, Tuple, Optional
import random
import datetime
import math
from collections import namedtuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a named tuple for PPO experience
PPOExperience = namedtuple('PPOExperience', ['src', 'tgt', 'actions', 'rewards', 'log_probs', 'values'])

class SelfEvaluationHead(nn.Module):
    """Self-evaluation head for RLAIF scoring"""
    def __init__(self, d_model: int):
        super(SelfEvaluationHead, self).__init__()
        self.eval_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, d_model] - pooled embeddings
        Returns:
            scores: [batch_size, 1] - quality scores 0-1
        """
        return self.eval_head(embeddings) * 5.0

class DiversityScorer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(DiversityScorer, self).__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.diversity_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vocab_embedding(token_ids)
        pooled = torch.mean(embeddings, dim=1)
        diversity_score = self.diversity_net(pooled)
        return diversity_score

class ComplexAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, max_context_length: int = 16384):
        super(ComplexAttention, self).__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.context_weight = nn.Linear(d_model, 1)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.max_context_length = max_context_length

    def generate_square_subsequent_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                context: Optional[torch.Tensor] = None, is_causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if query.size(1) > self.max_context_length:
                query = query[:, -self.max_context_length:]
                key = key[:, -self.max_context_length:]
                value = value[:, -self.max_context_length:]
            
            attn_mask = None
            if is_causal:
                seq_len = query.size(1)
                attn_mask = self.generate_square_subsequent_mask(seq_len, query.device)
            
            attn_output, attn_weights = self.attention(query, key, value, attn_mask=attn_mask)
            
            if context is not None:
                context_score = torch.sigmoid(self.context_weight(context))
                attn_output = attn_output * context_score
                attn_output, _ = self.cross_attention(attn_output, context, context)
            
            return attn_output, attn_weights
        except Exception as e:
            logging.error(f"Error in ComplexAttention forward: {e}")
            raise

class BatakTransformerPPO(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16, 
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12, 
                 dim_feedforward: int = 4096, max_seq_len: int = 16384):
        super(BatakTransformerPPO, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu"
        )
        self.complex_attention = ComplexAttention(d_model, nhead, max_seq_len)
        self.fc_out = nn.Linear(d_model, vocab_size) # Policy Head
        
        # Value Head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.self_eval_head = SelfEvaluationHead(d_model)
        self.diversity_scorer = DiversityScorer(vocab_size, d_model)
        
        self.transformer.encoder._use_gradient_checkpointing = True
        self.transformer.decoder._use_gradient_checkpointing = True

    def get_sinusoidal_pos_encoding(self, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        try:
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(seq_len, d_model, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)
        except Exception as e:
            logging.error(f"Error in get_sinusoidal_pos_encoding: {e}")
            raise

    def mean_pooling(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            return embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            return embeddings.mean(dim=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            
            src_emb = src_emb + self.get_sinusoidal_pos_encoding(src.size(1), self.d_model, src.device)
            tgt_emb = tgt_emb + self.get_sinusoidal_pos_encoding(tgt.size(1), self.d_model, tgt.device)
            
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            transformer_output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            attn_output, _ = self.complex_attention(transformer_output, transformer_output, 
                                                  transformer_output, context, is_causal=True)
            return self.fc_out(attn_output)
        except Exception as e:
            logging.error(f"Error in BatakTransformerPPO forward: {e}")
            raise
            
    def forward_policy_value(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            
            src_emb = src_emb + self.get_sinusoidal_pos_encoding(src.size(1), self.d_model, src.device)
            tgt_emb = tgt_emb + self.get_sinusoidal_pos_encoding(tgt.size(1), self.d_model, tgt.device)
            
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            transformer_output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            
            logits = self.fc_out(transformer_output)
            values = self.value_head(transformer_output)
            
            return logits, values
        except Exception as e:
            logging.error(f"Error in forward_policy_value: {e}")
            raise

    def self_evaluate(self, input_ids: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
        """Self-evaluation using internal evaluation head"""
        try:
            with torch.no_grad():
                # Pad/truncate to a consistent length before pooling if necessary
                max_len = max(input_ids.size(1), output_ids.size(1))
                input_ids_padded = F.pad(input_ids, (0, max_len - input_ids.size(1)), 'constant', 1)
                output_ids_padded = F.pad(output_ids, (0, max_len - output_ids.size(1)), 'constant', 1)
            
                input_emb = self.mean_pooling(self.embedding(input_ids_padded))
                output_emb = self.mean_pooling(self.embedding(output_ids_padded))
                
                # Simple concatenation
                combined_emb = torch.cat([input_emb, output_emb], dim=-1)
                # Ensure the linear layer size matches the concatenated embedding size
                combined_emb = nn.Linear(combined_emb.size(-1), self.d_model).to(combined_emb.device)(combined_emb)
                
                score = self.self_eval_head(combined_emb)
                return score
        except Exception as e:
            logging.error(f"Error in self_evaluate: {e}")
            return torch.tensor([2.5], device=input_ids.device)

    def calculate_reward_with_level(self, input_ids: torch.Tensor, output_ids: torch.Tensor, level: str) -> float:
        """Calculate final reward based on self-evaluation score and difficulty level."""
        try:
            base_score = self.self_evaluate(input_ids, output_ids).item()
            
            # Reward bonus based on level
            reward_bonus = {"easy": 0.0, "medium": 0.2, "hard": 0.5}.get(level.lower(), 0.0)
            
            final_reward = min(5.0, base_score + reward_bonus)
            return final_reward
        except Exception as e:
            logging.error(f"Error in calculate_reward_with_level: {e}")
            return 0.0

    def generate_multiple(self, src: torch.Tensor, context: Optional[torch.Tensor] = None, 
                         num_candidates: int = 3, max_len: int = 100, start_token_id: int = 2, 
                         sep_token_id: int = 3, temperature: float = 1.0) -> List[Tuple[torch.Tensor, float]]:
        """Generate multiple candidate responses with self-evaluation scores"""
        try:
            self.eval()
            candidates = []
            
            with torch.no_grad():
                src_emb = self.embedding(src) * math.sqrt(self.d_model)
                src_emb = src_emb + self.get_sinusoidal_pos_encoding(src.size(1), self.d_model, src.device)
                memory = self.transformer.encoder(src_emb)
                
                for _ in range(num_candidates):
                    tgt = torch.tensor([[start_token_id]], dtype=torch.long, device=src.device)
                    
                    for _ in range(max_len):
                        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
                        tgt_emb = tgt_emb + self.get_sinusoidal_pos_encoding(tgt.size(1), self.d_model, tgt.device)
                        
                        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
                        
                        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                        attn_output, _ = self.complex_attention(output, output, output, context, is_causal=True)
                        logits = self.fc_out(attn_output[:, -1, :]) / temperature
                        
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        tgt = torch.cat((tgt, next_token), dim=1)
                        
                        if next_token.item() == sep_token_id:
                            break
                    
                    score = self.self_evaluate(src, tgt).item()
                    candidates.append((tgt, score))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                
            return candidates
        except Exception as e:
            logging.error(f"Error in generate_multiple: {e}")
            return []
            
    def generate_ppo_experience(self, src: torch.Tensor, level: str, max_len: int = 100, start_token_id: int = 2, 
                                sep_token_id: int = 3, temperature: float = 1.0) -> Tuple[torch.Tensor, List[int], List[float], List[float]]:
        """Generate a single response and collect experience for PPO"""
        self.eval()
        with torch.no_grad():
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            src_emb = src_emb + self.get_sinusoidal_pos_encoding(src.size(1), self.d_model, src.device)
            memory = self.transformer.encoder(src_emb)
            
            actions = []
            log_probs = []
            values = []
            tgt = torch.tensor([[start_token_id]], dtype=torch.long, device=src.device)
            
            for _ in range(max_len):
                tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.get_sinusoidal_pos_encoding(tgt.size(1), self.d_model, tgt.device)
                
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
                
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Get logits and values from the last token
                last_output = output[:, -1, :]
                logits = self.fc_out(last_output) / temperature
                value = self.value_head(last_output)
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_token = dist.sample()
                log_prob = dist.log_prob(next_token)
                
                actions.append(next_token.item())
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                tgt = torch.cat((tgt, next_token.unsqueeze(0)), dim=1)
                
                if next_token.item() == sep_token_id:
                    break
            
            # Use the new reward function
            final_reward = self.calculate_reward_with_level(src, tgt, level)
            
            # Simple reward shaping: only final reward
            rewards = [0.0] * len(actions)
            if rewards:
                rewards[-1] = final_reward
                
            return tgt, actions, rewards, log_probs, values


class SmartAssistant:
    def __init__(self, data_folder: str = "batakdata", vocab_size: int = 50000, max_context_length: int = 16384):
        self.data_folder = data_folder
        self.finetune_folder = "finetuning"
        self.vocab_size = vocab_size
        self.normalizers = Sequence([NFKC(), Lowercase(), Strip()])
        self.max_context_length = max_context_length
        
        self.tokenizer = None
        self.model = None
        self.text_data: List[Dict] = []
        self.math_data: List[Dict] = []
        self.data_pairs: List[Dict] = []
        self.context_history: List[int] = []
        self.finetune_rules: Dict[str, List[str]] = {}
        self.vocabulary: set = set()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.eval_optimizer = None
        self.diversity_optimizer = None
        self.ppo_optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)
        
        self.ranking_buffer = []
        self.buffer_size = 100
        self.accumulation_steps = 4
        self.accumulated_loss = 0.0
        self.save_interval = 10
        self.input_count = 0
        
        self.ppo_experience_buffer = []
        self.ppo_epochs = 4
        self.ppo_clip_epsilon = 0.2
        self.ppo_value_coeff = 0.5
        self.ppo_entropy_coeff = 0.01

        self.log_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.training_logs = []
        
        self.pad_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        
        self.load_data()
        self.load_finetune_data()
        self.init_tokenizer()
        self.init_model()
        self.prepare_data()
        self.train_with_curriculum()

    def save_log(self, log_entry: Dict) -> None:
        try:
            self.training_logs.append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving log: {e}")

    def load_logs(self) -> List[Dict]:
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Error loading logs: {e}")
            return []

    def calculate_diversity_score(self, token_ids: List[int]) -> float:
        if not token_ids:
            return 0.0
        unique_tokens = len(set(token_ids))
        total_tokens = len(token_ids)
        return unique_tokens / total_tokens

    def add_to_ranking_buffer(self, input_text: str, candidates: List[Tuple[str, float]]) -> None:
        """Add ranked candidates to buffer for DPO training"""
        try:
            if len(candidates) < 2:
                return
                
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            chosen = candidates_sorted[0]
            rejected = candidates_sorted[-1]
            
            ranking_entry = {
                "prompt": input_text,
                "chosen": chosen[0],
                "rejected": rejected[0],
                "chosen_score": chosen[1],
                "rejected_score": rejected[1],
                "score_diff": chosen[1] - rejected[1],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.ranking_buffer.append(ranking_entry)
            
            if len(self.ranking_buffer) > self.buffer_size:
                self.ranking_buffer.pop(0)
            
            logging.info(f"Added ranking to buffer. Buffer size: {len(self.ranking_buffer)}")
            
        except Exception as e:
            logging.error(f"Error adding to ranking buffer: {e}")

    def dpo_loss(self, chosen_logits: torch.Tensor, rejected_logits: torch.Tensor, 
                 chosen_ids: torch.Tensor, rejected_ids: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """Direct Preference Optimization loss"""
        try:
            chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
            rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
            
            chosen_selected = torch.gather(chosen_log_probs, dim=-1, index=chosen_ids.unsqueeze(-1)).squeeze(-1)
            rejected_selected = torch.gather(rejected_log_probs, dim=-1, index=rejected_ids.unsqueeze(-1)).squeeze(-1)
            
            chosen_mask = (chosen_ids != self.pad_token_id).float()
            rejected_mask = (rejected_ids != self.pad_token_id).float()
            
            chosen_selected = chosen_selected * chosen_mask
            rejected_selected = rejected_selected * rejected_mask
            
            chosen_score = chosen_selected.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1e-5)
            rejected_score = rejected_selected.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1e-5)
            
            preference_diff = beta * (chosen_score - rejected_score)
            loss = -torch.log(torch.sigmoid(preference_diff)).mean()
            
            return loss
            
        except Exception as e:
            logging.error(f"Error in dpo_loss: {e}")
            return torch.tensor(0.0, device=chosen_logits.device)

    def train_dpo_step(self) -> float:
        """Training step using DPO on ranking buffer"""
        try:
            if len(self.ranking_buffer) < 2:
                return 0.0
                
            self.model.train()
            batch_loss = 0.0
            
            batch_data = random.sample(self.ranking_buffer, min(4, len(self.ranking_buffer)))
            
            for entry in batch_data:
                prompt_ids = self.tokenizer.encode(entry["prompt"]).ids
                chosen_ids = self.tokenizer.encode(entry["chosen"]).ids
                rejected_ids = self.tokenizer.encode(entry["rejected"]).ids
                
                max_len = max(len(prompt_ids), len(chosen_ids), len(rejected_ids))
                
                prompt_padded = prompt_ids + [self.pad_token_id] * (max_len - len(prompt_ids))
                chosen_padded = chosen_ids + [self.pad_token_id] * (max_len - len(chosen_ids))
                rejected_padded = rejected_ids + [self.pad_token_id] * (max_len - len(rejected_ids))
                
                prompt_tensor = torch.tensor([prompt_padded], dtype=torch.long, device=self.device)
                chosen_tensor = torch.tensor([chosen_padded], dtype=torch.long, device=self.device)
                rejected_tensor = torch.tensor([rejected_padded], dtype=torch.long, device=self.device)
                
                chosen_logits = self.model(prompt_tensor, chosen_tensor[:, :-1])
                rejected_logits = self.model(prompt_tensor, rejected_tensor[:, :-1])
                
                dpo_loss = self.dpo_loss(chosen_logits.view(-1, chosen_logits.size(-1)),
                                       rejected_logits.view(-1, rejected_logits.size(-1)),
                                       chosen_tensor[:, 1:].reshape(-1),
                                       rejected_tensor[:, 1:].reshape(-1))
                
                self.optimizer.zero_grad()
                dpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                batch_loss += dpo_loss.item()
            
            avg_loss = batch_loss / len(batch_data)
            
            dpo_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "dpo_training",
                "batch_size": len(batch_data),
                "average_loss": avg_loss,
                "buffer_size": len(self.ranking_buffer)
            }
            self.save_log(dpo_log)
            
            return avg_loss
            
        except Exception as e:
            logging.error(f"Error in train_dpo_step: {e}")
            return 0.0

    def train_self_eval_head(self, epochs: int = 5) -> None:
        """Train self-evaluation head using ranking buffer data with real scores"""
        try:
            if not self.ranking_buffer:
                logging.warning("No ranking buffer data available for self-evaluation head training")
                return
                
            self.model.self_eval_head.train()
            eval_criterion = nn.MSELoss()
            
            logging.info(f"Training self-evaluation head with {len(self.ranking_buffer)} ranking buffer entries")
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                for entry in random.sample(self.ranking_buffer, min(50, len(self.ranking_buffer))):
                    input_tensor_ids = self.tokenizer.encode(entry["prompt"]).ids
                    chosen_tensor_ids = self.tokenizer.encode(entry["chosen"]).ids
                    rejected_tensor_ids = self.tokenizer.encode(entry["rejected"]).ids
                    
                    input_tensor = torch.tensor([input_tensor_ids], dtype=torch.long, device=self.device)
                    chosen_tensor = torch.tensor([chosen_tensor_ids], dtype=torch.long, device=self.device)
                    rejected_tensor = torch.tensor([rejected_tensor_ids], dtype=torch.long, device=self.device)
                    
                    chosen_score_target = torch.tensor([[entry["chosen_score"]]], dtype=torch.float, device=self.device)
                    rejected_score_target = torch.tensor([[entry["rejected_score"]]], dtype=torch.float, device=self.device)
                    
                    self.eval_optimizer.zero_grad()
                    
                    pred_chosen = self.model.self_evaluate(input_tensor, chosen_tensor)
                    pred_rejected = self.model.self_evaluate(input_tensor, rejected_tensor)
                    
                    loss = eval_criterion(pred_chosen, chosen_score_target) + eval_criterion(pred_rejected, rejected_score_target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.self_eval_head.parameters(), max_norm=1.0)
                    self.eval_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                logging.info(f"Self-eval head training epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                
                eval_log = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "self_eval_training",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "average_loss": avg_loss,
                    "num_samples": num_batches
                }
                self.save_log(eval_log)
                
        except Exception as e:
            logging.error(f"Error in train_self_eval_head: {e}")

    def ppo_loss(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor,
                 values: torch.Tensor, rewards: torch.Tensor,
                 gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            # Calculate advantage using GAE
            advantages = []
            last_advantage = 0
            # Ensure rewards and values have the same length
            rewards = rewards[:len(values)-1]
            
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t+1] - values[t] if t < len(rewards) - 1 else rewards[t] - values[t]
                last_advantage = delta + gamma * gae_lambda * last_advantage
                advantages.insert(0, last_advantage)
            
            advantages = torch.tensor(advantages, dtype=torch.float, device=self.device).detach()
            returns = advantages + values[:-1].detach()
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values[:-1].squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -new_log_probs.mean()
            
            return policy_loss, value_loss, entropy_loss

        except Exception as e:
            logging.error(f"Error in ppo_loss: {e}")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def train_ppo_step(self):
        try:
            if not self.ppo_experience_buffer:
                logging.warning("PPO experience buffer is empty.")
                return 0.0

            self.model.train()
            total_loss = 0.0

            for _ in range(self.ppo_epochs):
                for experience in self.ppo_experience_buffer:
                    src_tensor = experience.src
                    tgt_tensor = experience.tgt
                    old_log_probs = torch.tensor(experience.log_probs, device=self.device)
                    rewards = torch.tensor(experience.rewards, device=self.device)
                    values = torch.tensor(experience.values, device=self.device)

                    self.ppo_optimizer.zero_grad()

                    logits, new_values = self.model.forward_policy_value(src_tensor, tgt_tensor[:, :-1])
                    new_log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Gather new log_probs for the taken actions
                    new_log_probs_selected = torch.gather(new_log_probs.view(-1, self.model.vocab_size), 1, tgt_tensor[:, 1:].reshape(-1).unsqueeze(1)).squeeze(1)

                    policy_loss, value_loss, entropy_loss = self.ppo_loss(
                        old_log_probs, new_log_probs_selected, 
                        torch.cat([new_values.squeeze(), torch.tensor([0.0], device=self.device)]), 
                        rewards
                    )

                    loss = policy_loss + self.ppo_value_coeff * value_loss + self.ppo_entropy_coeff * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.ppo_optimizer.step()
                    
                    total_loss += loss.item()

            avg_loss = total_loss / (len(self.ppo_experience_buffer) * self.ppo_epochs)
            
            ppo_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "ppo_training",
                "average_loss": avg_loss,
                "buffer_size": len(self.ppo_experience_buffer)
            }
            self.save_log(ppo_log)
            
            # Clear buffer after training
            self.ppo_experience_buffer.clear()
            
            return avg_loss

        except Exception as e:
            logging.error(f"Error in train_ppo_step: {e}")
            return 0.0


    def load_finetune_data(self) -> None:
        try:
            if not os.path.exists(self.finetune_folder):
                logging.warning(f"Fine-tuning folder {self.finetune_folder} does not exist")
                return
                
            for i in range(1, 1000):
                json_file_path = os.path.join(self.finetune_folder, f"finebatak{i}.json")
                if not os.path.exists(json_file_path):
                    break
                    
                try:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        finetune_data = json.load(f)
                        for item in finetune_data:
                            for key, value in item.items():
                                if isinstance(value, list):
                                    self.finetune_rules[key.lower()] = [v.lower() for v in value]
                    logging.info(f"Loaded fine-tuning data from {json_file_path}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON in {json_file_path}: {e}")
                except Exception as e:
                    logging.error(f"Error loading fine-tuning data from {json_file_path}: {e}")
                    
            if not self.finetune_rules:
                logging.info("No fine-tuning rules loaded")
            else:
                logging.info(f"Loaded {len(self.finetune_rules)} fine-tuning rules")
        except Exception as e:
            logging.error(f"Error in load_finetune_data: {e}")

    def apply_finetune_rules(self, text: str) -> str:
        try:
            modified_text = text.lower()
            for original, replacements in self.finetune_rules.items():
                if replacements:
                    replacement = np.random.choice(replacements)
                    modified_text = modified_text.replace(original, replacement)
            return modified_text
        except Exception as e:
            logging.error(f"Error applying fine-tuning rules: {e}")
            return text

    def load_data(self) -> None:
        self.text_data = []
        self.math_data = []
        self.vocabulary = set()
        
        if not os.path.exists(self.data_folder):
            logging.error(f"Data folder {self.data_folder} does not exist")
            raise FileNotFoundError(f"Data folder {self.data_folder} not found")
        
        found_data = False
        
        for i in range(1, 1000):
            txt_file_path = os.path.join(self.data_folder, f"batakdata{i}.txt")
            if os.path.exists(txt_file_path):
                try:
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "|" in line:
                                try:
                                    parts = line.strip().split("|")
                                    a = self.apply_finetune_rules(parts[0].replace("A:", "").strip())
                                    b = self.apply_finetune_rules(parts[1].replace("B:", "").strip())
                                    level = parts[2].replace("level:", "").strip().lower() if len(parts) > 2 and parts[2].replace("level:", "").strip().lower() in ["easy", "medium", "hard"] else "easy"
                                    if a and b:
                                        self.text_data.append({"prompt": a, "response": b, "level": level})
                                        self.vocabulary.update(a.split())
                                        self.vocabulary.update(b.split())
                                        found_data = True
                                except ValueError:
                                    logging.warning(f"Invalid line format in {txt_file_path}: {line.strip()}")
                except Exception as e:
                    logging.error(f"Failed to read {txt_file_path}: {e}")
                    continue
            
            json_file_path = os.path.join(self.data_folder, f"batakdata{i}.json")
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        math_data_list = json.load(f)
                        for item in math_data_list:
                            if all(key in item for key in ["question", "answer", "reasoning", "level"]):
                                if all(isinstance(item[key], str) for key in ["question", "answer", "reasoning", "level"]):
                                    item["question"] = self.apply_finetune_rules(item["question"])
                                    item["answer"] = self.apply_finetune_rules(item["answer"])
                                    item["reasoning"] = self.apply_finetune_rules(item["reasoning"])
                                    item["level"] = item["level"].lower() if item["level"].lower() in ["easy", "medium", "hard"] else "easy"
                                    self.math_data.append(item)
                                    self.vocabulary.update(item["question"].split())
                                    self.vocabulary.update(item["reasoning"].split())
                                    found_data = True
                                else:
                                    logging.warning(f"Invalid math data format in {json_file_path}: {item}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON in {json_file_path}: {e}")
                    continue
            
            if not (os.path.exists(txt_file_path) or os.path.exists(json_file_path)):
                break
        
        if not found_data or (not self.text_data and not self.math_data):
            logging.error("No valid linguistic or math data loaded")
            raise ValueError("Both linguistic and math datasets must be non-empty")
        
        logging.info(f"Loaded {len(self.text_data)} text pairs, {len(self.math_data)} math problems, and {len(self.vocabulary)} unique words")

    def init_tokenizer(self) -> None:
        try:
            self.tokenizer = Tokenizer(Unigram())
            self.tokenizer.normalizer = self.normalizers
            self.tokenizer.pre_tokenizer = ByteLevel()
            self.tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B [SEP]",
                special_tokens=[("[UNK]", 0), ("[PAD]", 1), ("[CLS]", 2), ("[SEP]", 3), ("[MASK]", 4), ("[LEVEL:easy]", 5), ("[LEVEL:medium]", 6), ("[LEVEL:hard]", 7)]
            )
            
            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[LEVEL:easy]", "[LEVEL:medium]", "[LEVEL:hard]"],
                unk_token="[UNK]"
            )
            
            all_sentences = ([item["prompt"] for item in self.text_data] + 
                           [item["response"] for item in self.text_data] + 
                           [item["question"] for item in self.math_data] + 
                           [item["answer"] for item in self.math_data] + 
                           [item["reasoning"] for item in self.math_data])
            
            if len(all_sentences) < 5000:
                logging.error("Insufficient data for tokenizer training: minimum 5000 entries required")
                # raise ValueError("Tokenizer requires at least 5000 data entries") # Temporarily commented out for testing
            
            self.tokenizer.train_from_iterator(all_sentences, trainer)
            self.tokenizer.save("bataknese_tokenizer.json")
            logging.info(f"Tokenizer trained with {len(all_sentences)} sentences")
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer: {e}")
            raise

    def init_model(self) -> None:
        try:
            vocab_size = len(self.tokenizer.get_vocab())
            if vocab_size < 1000:
                raise ValueError(f"Vocabulary size too small: {vocab_size}")
            
            self.model = BatakTransformerPPO(vocab_size=vocab_size).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) # For DPO & Supervised
            self.eval_optimizer = torch.optim.Adam(self.model.self_eval_head.parameters(), lr=1e-4)
            self.diversity_optimizer = torch.optim.Adam(self.model.diversity_scorer.parameters(), lr=1e-4)
            self.ppo_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) # For PPO
            
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"Model initialized with {param_count} parameters")
            
            if param_count < 100_000_000:
                logging.warning(f"Model size ({param_count}) is below 100M parameters; consider increasing layers or dimensions")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise

    def prepare_data(self) -> None:
        self.data_pairs = []
        
        all_data = self.text_data + self.math_data
        
        for item in all_data:
            try:
                if "prompt" in item:
                    input_text = item["prompt"]
                    target_text = item["response"]
                else: # math data
                    input_text = item["question"]
                    target_text = f"Penjelasan: {item['reasoning']} Jawaban: {item['answer']}"
                
                # Inject level token into the prompt
                level = item.get("level", "easy")
                level_token_text = f"[LEVEL:{level}]"
                
                input_encoding = self.tokenizer.encode(f"{level_token_text} {input_text}")
                target_encoding = self.tokenizer.encode(target_text)
                
                input_ids = input_encoding.ids
                target_ids = target_encoding.ids
                
                if len(input_ids) > self.max_context_length:
                    input_ids = input_ids[-self.max_context_length:]
                if len(target_ids) > self.max_context_length:
                    target_ids = target_ids[-self.max_context_length:]
                
                self.data_pairs.append({
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "input_tokens": input_encoding.tokens,
                    "target_tokens": target_encoding.tokens,
                    "level": level
                })
            except Exception as e:
                logging.warning(f"Failed to encode sentence pair for prompt '{input_text}': {e}")
        
        if not self.data_pairs:
            logging.error("No valid linguistic data pairs prepared")
            raise ValueError("Linguistic data pairs could not be prepared")
        
        logging.info(f"Prepared {len(self.data_pairs)} data pairs for training")

    def train_with_curriculum(self, epochs: int = 5, batch_size: int = 8) -> None:
        try:
            self.model.train()
            easy_data = [item for item in self.data_pairs if item.get("level") == "easy"]
            medium_data = [item for item in self.data_pairs if item.get("level") == "medium"]
            hard_data = [item for item in self.data_pairs if item.get("level") == "hard"]
            
            training_schedule = [
                (easy_data, "EASY"),
                (easy_data + medium_data, "EASY + MEDIUM"),
                (easy_data + medium_data + hard_data, "ALL")
            ]
            
            for epoch in range(epochs):
                if epoch < epochs // 3:
                    current_training_data, level_desc = training_schedule[0]
                elif epoch < (epochs * 2) // 3:
                    current_training_data, level_desc = training_schedule[1]
                else:
                    current_training_data, level_desc = training_schedule[2]
                
                if not current_training_data:
                    logging.warning(f"No data available for {level_desc} level training. Skipping epoch {epoch + 1}.")
                    continue
                
                logging.info(f"Epoch {epoch + 1}/{epochs}: Training with {level_desc} data. Samples: {len(current_training_data)}")
                
                random.shuffle(current_training_data)
                
                total_loss = 0.0
                num_batches = 0
                
                for i in range(0, len(current_training_data), batch_size):
                    batch = current_training_data[i:i + batch_size]
                    
                    max_input_len = max(len(pair["input_ids"]) for pair in batch)
                    max_target_len = max(len(pair["target_ids"]) for pair in batch)
                    
                    input_tensors = []
                    target_tensors = []
                    
                    for pair in batch:
                        input_ids = pair["input_ids"] + [self.pad_token_id] * (max_input_len - len(pair["input_ids"]))
                        target_ids = pair["target_ids"] + [self.pad_token_id] * (max_target_len - len(pair["target_ids"]))
                        
                        input_tensors.append(input_ids)
                        target_tensors.append(target_ids)
                    
                    input_tensors = torch.tensor(input_tensors, dtype=torch.long, device=self.device)
                    target_tensors = torch.tensor(target_tensors, dtype=torch.long, device=self.device)
                    
                    self.optimizer.zero_grad()
                    
                    output = self.model(input_tensors, target_tensors[:, :-1])
                    loss = self.criterion(output.view(-1, output.size(-1)), target_tensors[:, 1:].reshape(-1))
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                logging.info(f"Training epoch {epoch + 1}/{epochs}, Average loss: {avg_loss:.4f}")
                
                train_log = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "curriculum_training",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "average_loss": avg_loss,
                    "level_trained": level_desc
                }
                self.save_log(train_log)
            
            self.save_model()
        except Exception as e:
            logging.error(f"Error in train_with_curriculum: {e}")

    def update_context(self, input_ids: List[int]) -> None:
        try:
            self.context_history.extend(input_ids)
            if len(self.context_history) > self.max_context_length:
                self.context_history = self.context_history[-self.max_context_length:]
        except Exception as e:
            logging.error(f"Error in update_context: {e}")

    def classify_question_difficulty(self, text: str) -> str:
        """Classifies a question as easy, medium, or hard based on simple heuristics."""
        try:
            words = text.lower().split()
            num_words = len(words)
            
            hard_keywords = ["mengapa", "analisis", "bagaimana", "bandingkan", "penjelasan rinci", "jelaskan konsekuensi"]
            medium_keywords = ["sebab", "efek", "contoh", "peran", "hubungan", "tentang", "fakta", "ciri-ciri"]
            
            has_hard_keyword = any(keyword in text.lower() for keyword in hard_keywords)
            has_medium_keyword = any(keyword in text.lower() for keyword in medium_keywords)
            
            if has_hard_keyword or num_words > 30:
                return "hard"
            elif has_medium_keyword or num_words > 15:
                return "medium"
            else:
                return "easy"
        except Exception as e:
            logging.error(f"Error classifying question difficulty: {e}")
            return "easy"

    def retrieve_relevant_context(self, input_ids: List[int]) -> List[int]:
        try:
            if not self.data_pairs:
                return self.context_history[-self.max_context_length:]
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                input_emb = self.model.mean_pooling(self.model.embedding(input_tensor))
                
                best_score = -1.0
                best_context = None
                
                for pair in self.data_pairs[:100]:
                    context_tensor = torch.tensor(pair["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
                    context_emb = self.model.mean_pooling(self.model.embedding(context_tensor))
                    
                    cosine_sim = F.cosine_similarity(input_emb, context_emb).item()
                    
                    if cosine_sim > best_score:
                        best_score = cosine_sim
                        best_context = pair["target_ids"]
                
                context_ids = self.context_history[-self.max_context_length//2:] + (best_context or [])
                
                if len(context_ids) > self.max_context_length:
                    context_ids = context_ids[-self.max_context_length:]
                
                return context_ids
        except Exception as e:
            logging.error(f"Error in retrieve_relevant_context: {e}")
            return []

    def save_model(self, save_path: str = "batak_model.pth") -> None:
        try:
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'eval_optimizer_state_dict': self.eval_optimizer.state_dict(),
                'diversity_optimizer_state_dict': self.diversity_optimizer.state_dict(),
                'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
                'vocab_size': self.vocab_size,
                'max_context_length': self.max_context_length,
                'ranking_buffer': self.ranking_buffer,
                'ppo_experience_buffer': self.ppo_experience_buffer
            }
            
            torch.save(model_state, save_path)
            self.tokenizer.save("bataknese_tokenizer.json")
            
            logging.info(f"Model and optimizers saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, load_path: str = "batak_model.pth") -> None:
        try:
            if os.path.exists(load_path):
                checkpoint = torch.load(load_path, map_location=self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.eval_optimizer.load_state_dict(checkpoint['eval_optimizer_state_dict'])
                self.diversity_optimizer.load_state_dict(checkpoint['diversity_optimizer_state_dict'])
                
                if 'ppo_optimizer_state_dict' in checkpoint:
                    self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])

                if 'ranking_buffer' in checkpoint:
                    self.ranking_buffer = checkpoint['ranking_buffer']
                
                if 'ppo_experience_buffer' in checkpoint:
                    self.ppo_experience_buffer = checkpoint['ppo_experience_buffer']
                
                logging.info(f"Model and optimizers loaded from {load_path}")
            else:
                logging.warning(f"Model file {load_path} not found")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def predict(self, input_text: str, user_score: Optional[int] = None) -> Tuple[Optional[str], Optional[List[str]], Optional[List[int]]]:
        try:
            self.input_count += 1
            
            if len(input_text.strip()) < 3:
                logging.warning("Input too short")
                return "Input terlalu pendek. Minimal 3 karakter.", None, None
            
            # Classify the difficulty level of the new input
            input_level = self.classify_question_difficulty(input_text)
            logging.info(f"Input level classified: {input_level}")
            
            # Inject level token into the input text
            input_text_with_level = f"[LEVEL:{input_level}] {input_text}"
            
            encoding = self.tokenizer.encode(self.apply_finetune_rules(input_text_with_level))
            input_ids = encoding.ids
            
            if not input_ids:
                logging.error("Empty input encoding")
                return None, None, None
            
            self.update_context(input_ids)
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            context_ids = self.retrieve_relevant_context(input_ids)
            context_tensor = torch.tensor([context_ids], dtype=torch.long, device=self.device) if context_ids else None
            
            # Set generation parameters based on level
            temperature_map = {"easy": 0.8, "medium": 1.0, "hard": 1.2}
            generation_temperature = temperature_map.get(input_level, 1.0)
            
            # First pass for best candidate
            candidates = self.model.generate_multiple(
                input_tensor, 
                context_tensor, 
                num_candidates=3,
                max_len=100, 
                start_token_id=self.cls_token_id,
                sep_token_id=self.sep_token_id,
                temperature=generation_temperature
            )
            
            if not candidates:
                return "Maaf, tidak dapat menghasilkan respons.", None, None

            best_candidate = candidates[0]
            best_score = best_candidate[1]
            
            self_score_threshold = 3.5  # Ambang batas skor untuk eksplorasi ulang
            
            # Uncertainty Feedback Loop: check if score is low, especially for harder questions
            if best_score < self_score_threshold and input_level in ["medium", "hard"]:
                logging.info(f"Uncertainty detected for {input_level} question: self_score ({best_score:.2f}) < threshold ({self_score_threshold}). Re-sampling with higher temperature.")
                candidates_exploration = self.model.generate_multiple(
                    input_tensor,
                    context_tensor,
                    num_candidates=5,
                    max_len=100,
                    start_token_id=self.cls_token_id,
                    sep_token_id=self.sep_token_id,
                    temperature=1.5 # higher temperature for exploration
                )
                if candidates_exploration:
                    candidates = candidates_exploration
                    best_candidate = candidates[0]
                    best_score = best_candidate[1]
            
            # Use PPO generation to collect experience
            response_ids, actions, rewards, log_probs, values = self.model.generate_ppo_experience(
                input_tensor,
                level=input_level,
                max_len=100, 
                start_token_id=self.cls_token_id,
                sep_token_id=self.sep_token_id,
                temperature=generation_temperature
            )
            
            # Store the PPO experience
            self.ppo_experience_buffer.append(PPOExperience(
                src=input_tensor,
                tgt=response_ids,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values
            ))

            best_output_ids = best_candidate[0][0].cpu().numpy()
            best_output_text = self.tokenizer.decode(best_output_ids)
            best_output_text = best_output_text.replace("[CLS]", "").replace("[SEP]", "").strip()
            
            if "berapa " in input_text.lower() or "+" in input_text or "-" in input_text: # Simple check for math
                best_output_text = f"Penjelasan: Ini adalah pertanyaan matematika yang memerlukan analisis neural.\nJawaban: {best_output_text}"
            
            candidate_texts = []
            for candidate, score in candidates:
                output_ids = candidate[0].cpu().numpy()
                output_text = self.tokenizer.decode(output_ids)
                output_text = output_text.replace("[CLS]", "").replace("[SEP]", "").strip()
                candidate_texts.append((output_text, score))
            
            self.add_to_ranking_buffer(input_text_with_level, candidate_texts)
            
            # Periodic training from buffer
            if self.input_count % 5 == 0:
                if len(self.ppo_experience_buffer) >= 5:
                    ppo_loss_val = self.train_ppo_step()
                    logging.info(f"PPO training loss: {ppo_loss_val:.4f}")
                
                if len(self.ranking_buffer) >= 5:
                    dpo_loss = self.train_dpo_step()
                    logging.info(f"DPO training loss: {dpo_loss:.4f}")
            
            if self.input_count % 20 == 0:
                self.train_self_eval_head(epochs=2)
                logging.info("Self-evaluation head retrained")
            
            if self.input_count % self.save_interval == 0:
                self.save_model()
            
            prediction_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "rlaif_prediction",
                "input_text": input_text,
                "best_output": best_output_text,
                "best_score": best_score,
                "num_candidates": len(candidates),
                "candidate_scores": [score for _, score in candidates],
                "user_score": user_score,
                "input_count": self.input_count,
                "detected_level": input_level
            }
            self.save_log(prediction_log)
            
            return best_output_text, encoding.tokens, context_ids
            
        except Exception as e:
            logging.error(f"Error in predict for input '{input_text}': {e}")
            return None, None, None

    def process_feedback(self, input_text: str, correct_output: str, user_score: Optional[int] = None) -> None:
        try:
            input_level = self.classify_question_difficulty(input_text)
            input_text_with_level = f"[LEVEL:{input_level}] {input_text}"
            
            input_ids = self.tokenizer.encode(self.apply_finetune_rules(input_text_with_level)).ids
            target_ids = self.tokenizer.encode(self.apply_finetune_rules(correct_output)).ids
            
            max_len = max(len(input_ids), len(target_ids))
            input_ids_padded = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            target_ids_padded = target_ids + [self.pad_token_id] * (max_len - len(target_ids))
            
            input_tensor = torch.tensor([input_ids_padded], dtype=torch.long, device=self.device)
            target_tensor = torch.tensor([target_ids_padded], dtype=torch.long, device=self.device)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(input_tensor, target_tensor[:, :-1])
            loss = self.criterion(output.view(-1, output.size(-1)), target_tensor[:, 1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if user_score:
                feedback_entry = (correct_output, float(user_score))
                dummy_rejected_score = max(1.0, float(user_score) - 1.0)
                self.add_to_ranking_buffer(input_text_with_level, [(correct_output, float(user_score)), ("dummy rejected", dummy_rejected_score)])
            
            feedback_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "feedback_processing",
                "input_text": input_text,
                "correct_output": correct_output,
                "user_score": user_score,
                "supervised_loss": loss.item()
            }
            self.save_log(feedback_log)
            
            logging.info(f"Feedback processed - Loss: {loss.item():.4f}")
        except Exception as e:
            logging.error(f"Error processing feedback: {e}")

def main():
    try:
        assistant = SmartAssistant()
        print("Selamat datang di Enhanced RLAIF Batak LLM SmartAssistant!")
        print("Sistem menggunakan Self-Evaluation Head, DPO, dan PPO untuk RLAIF training.")
        print("Ketik kalimat dalam bahasa Batak atau soal matematika untuk mendapatkan respons.")
        print("Jika ingin memberikan feedback, ketik 'feedback:<input>:<correct_output>[:<score>]'. Skor opsional (1-5).")
        print("Ketik 'logs' untuk melihat log training terakhir.")
        print("Ketik 'exit' untuk keluar.")
        
        while True:
            user_input = input("\nInput: ")
            
            if user_input.lower() == "exit":
                print("Sampai jumpa lagi!")
                break
            
            if user_input.lower() == "logs":
                try:
                    logs = assistant.load_logs()
                    if logs:
                        print(f"Menampilkan {min(5, len(logs))} log terakhir:")
                        for log in logs[-5:]:
                            print(f"[{log['timestamp']}] {log['type']}: {log}")
                    else:
                        print("Tidak ada log yang tersedia.")
                except Exception as e:
                    print(f"Error loading logs: {e}")
                continue
            
            if not user_input.strip():
                print("Input tidak boleh kosong. Coba lagi.")
                continue
            
            if user_input.startswith("feedback:"):
                try:
                    parts = user_input.split(":")
                    if len(parts) >= 3:
                        if len(parts) == 4:
                            _, input_text, correct_output, score = parts
                            score = int(score) if score.isdigit() and 1 <= int(score) <= 5 else None
                        else:
                            _, input_text, correct_output = parts
                            score = None
                        
                        assistant.process_feedback(input_text, correct_output, score)
                        print(f"✓ Feedback diterima dan model diperbarui untuk input: {input_text}")
                    else:
                        print("Format feedback salah. Gunakan 'feedback:<input>:<correct_output>[:<score>]'.")
                except ValueError:
                    print("Format feedback salah. Gunakan 'feedback:<input>:<correct_output>[:<score>]'.")
                continue
            
            user_score = None
            if ":" in user_input and not user_input.startswith("feedback:"):
                try:
                    input_text, score = user_input.rsplit(":", 1)
                    user_score = int(score) if score.isdigit() and 1 <= int(score) <= 5 else None
                except ValueError:
                    input_text = user_input
            else:
                input_text = user_input
            
            output_text, input_tokens, context_ids = assistant.predict(input_text, user_score)
            
            if output_text:
                print(f"\n📝 Input Tokens: {input_tokens[:10]}...")
                print(f"🤖 Output: {output_text}")
                if context_ids:
                    print(f"🔍 Context Retrieved: {len(context_ids)} tokens")
                if user_score:
                    print(f"⭐ User Score: {user_score}/5")
                print(f"📊 Ranking Buffer Size: {len(assistant.ranking_buffer)}")
                print(f"📊 PPO Experience Buffer Size: {len(assistant.ppo_experience_buffer)}")
            else:
                print("❌ Gagal menghasilkan output. Coba lagi dengan kalimat atau soal lain.")
                
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()