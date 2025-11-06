"""
Continue training from checkpoint
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

# Import the model from bigger.py
import sys
sys.path.append('.')
from bigger import TRM10M, Config, WikiTextDataset

# Create output directories
OUTPUTS_DIR = Path("outputs")
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"

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
        
        if batch_idx % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            msg = f"Epoch {epoch} | Batch {batch_idx} | Loss: {avg_loss:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
        
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
    print("🔄 Continuing TRM Training from Checkpoint")
    print("=" * 50)
    
    # Configuration
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Load checkpoint
    checkpoint_path = "outputs/checkpoints/checkpoint_epoch9_batch28000.pt"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
    
    # Load existing results
    results_path = OUTPUTS_DIR / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previous results")
    else:
        results = []
    
    # Training log
    log_path = LOGS_DIR / f"training_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, 'w')
    
    # Find best perplexity so far
    best_val_ppl = min([r['val_perplexity'] for r in results]) if results else float('inf')
    print(f"Current best perplexity: {best_val_ppl:.2f}")
    
    print("\n" + "=" * 50)
    print("Continuing training...")
    print("=" * 50 + "\n")
    
    # Continue training for 10 more epochs (epochs 10-19)
    total_epochs = start_epoch + 10
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
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
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    log_file.close()
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print("=" * 50)

if __name__ == '__main__':
    main()

