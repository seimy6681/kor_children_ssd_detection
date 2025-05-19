import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Processor
from models.wav2vec2_encoder import Wav2Vec2Encoder
from utils.losses import TripletLoss
import wandb
from tqdm import tqdm
from utils.validation import validate
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import StepLR

def train_model(dataloader, train_df, valid_df, device, config):
    
    best_uar = 0.0
    best_thresholds = None
    
    model = Wav2Vec2Encoder().to(device)
    model.device = device
    model.wav2vec2.gradient_checkpointing_enable()

    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size =10, gamma=0.9)
    
    global_step = 0
    total_steps = config.epochs * len(dataloader)
    for epoch in range(config.epochs):
        
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)
        for batch in progress_bar:
            input_values = batch['input_values'].to(device)
            pos_values = batch['pos_values'].to(device)
            neg_values = batch['neg_values'].to(device)

            attention_mask = batch['attention_mask'].to(device)
            pos_mask = batch['pos_mask'].to(device)
            neg_mask = batch['neg_mask'].to(device)

            # memory saving
            with torch.cuda.amp.autocast():
                anchor_embed = model(input_values, attention_mask)
                pos_embed = model(pos_values, pos_mask)
                neg_embed = model(neg_values, neg_mask)

                loss, pos_distance, neg_distance = criterion(anchor_embed, pos_embed, neg_embed)
                
            # without memory saving
            # anchor_embed = model(input_values, attention_mask)
            # pos_embed = model(pos_values, pos_mask)
            # neg_embed = model(neg_values, neg_mask)
            # loss = criterion(anchor_embed, pos_embed, neg_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step_loss = loss.item()
            total_loss += step_loss
            
            global_step += 1
            progress_bar.set_postfix(step_loss=step_loss, step=f"{global_step}/{total_steps}")
            if config.wandb:
                wandb.log({
                        "step_loss": step_loss, 
                        "global_step": global_step,
                        "pos_distance:" :pos_distance,
                        "neg_distance": neg_distance
                        })
                
            # print(f"Step loss: {step_loss:.4f}, step {global_step}/")

        avg_loss = total_loss / (len(dataloader))
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")
        
        if config.wandb:
            wandb.log({"epoch": epoch+1, "epoch_loss": avg_loss})
            
        if config.run_inference:
            model.wav2vec2.gradient_checkpointing_disable()
            model.eval()

            val_acc, val_uar, thresholds, train_ref_embeddings = validate(model, train_df, valid_df, config)
            
            print(f"valid_acc: {val_acc:.4f}, valid_uar: {val_uar:.4f}")
            if config.wandb:
                wandb.log({
                        "validation/acc": val_acc,
                        "validation/uar: ": val_uar,
                          })

            if val_uar > best_uar:
                best_uar = val_uar
                best_thresholds = thresholds
                torch.save(model.state_dict(), config.model_path) # saving checkpoint with best uar
        
        model.train()
        model.wav2vec2.gradient_checkpointing_enable()    
        
    if config.wandb:      
        wandb.finish()
        
    torch.save(model.state_dict(), f'{config.model_path}')
    
    return best_thresholds, train_ref_embeddings


