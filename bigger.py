"""
TRM 10M Parameter Scaling Experiment
Conservative scaling to validate TRM approach
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Create output directories
OUTPUTS_DIR = Path("outputs")
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
for d in [OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# Configuration - Scaled up from proven tiny version
class Config:
    # Model architecture (targeting ~10M params)
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 6
    num_heads = 8
    vocab_size = 50257  # GPT-2 tokenizer
    
    # TRM specific - keeping proven hyperparameters
    num_refinements = 2
    recursion_depth = 3
    total_passes = 2 + (num_refinements * recursion_depth)  # 8 passes
    
    # Training - keeping proven learning rate
    learning_rate = 1e-4  # CRITICAL: Don't change this!
    batch_size = 8  # Adjust based on GPU memory
    seq_length = 512
    epochs = 10
    
    # Logging
    log_interval = 100
    eval_interval = 500
    save_interval = 1000

class TRMBlock(nn.Module):
    """Single TRM refinement block"""
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Dropout(0.1)
        )
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TRM10M(nn.Module):
    """10M parameter Token Refinement Machine"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_dim)
        
        # Base transformer layers
        self.base_layers = nn.ModuleList([
            TRMBlock(config) for _ in range(config.num_layers)
        ])
        
        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            TRMBlock(config) for _ in range(config.num_refinements)
        ])
        
        # Output projection
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.token_embedding.weight
        
        print(f"Model parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Initial embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Base pass through transformer
        for layer in self.base_layers:
            x = layer(x, mask)
        
        # Initial predictions
        logits = self.output(x)
        
        # Refinement passes with recursion
        for refinement_layer in self.refinement_layers:
            for _ in range(self.config.recursion_depth):
                # Use previous predictions as additional context
                refined = refinement_layer(x, mask)
                x = x + refined  # Residual connection
        
        # Final predictions
        logits = self.output(x)
        return logits

class WikiTextDataset(Dataset):
    """WikiText-103 dataset wrapper"""
    def __init__(self, split, tokenizer, seq_length):
        print(f"Loading WikiText-103 {split}...")
        self.data = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize and concatenate
        print("Tokenizing...")
        all_tokens = []
        for item in self.data:
            if item['text'].strip():
                tokens = tokenizer.encode(item['text'])
                all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens)
        print(f"Total tokens: {len(self.tokens):,}")
    
    def __len__(self):
        return len(self.tokens) // self.seq_length
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]  # input, target

def train_epoch(model, dataloader, optimizer, device, config, epoch, log_file):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            msg = f"Epoch {epoch} | Batch {batch_idx} | Loss: {avg_loss:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
        
        # Save checkpoint
        if batch_idx % config.save_interval == 0 and batch_idx > 0:
            checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1)
        )
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def main():
    print("🚀 TRM 10M Parameter Scaling Experiment")
    print("=" * 50)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = WikiTextDataset('train', tokenizer, config.seq_length)
    val_dataset = WikiTextDataset('validation', tokenizer, config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model
    print("\nInitializing TRM model...")
    model = TRM10M(config).to(device)
    
    # Optimizer - using proven learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training log
    log_path = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, 'w')
    
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")
    
    best_val_ppl = float('inf')
    results = []
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, log_file)
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, device, config)
        
        result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_ppl
        }
        results.append(result)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_ppl:.2f}")
        
        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pt")
            print(f"  ✓ New best perplexity: {val_ppl:.2f}")
        
        # Save results
        with open(OUTPUTS_DIR / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    log_file.close()
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
