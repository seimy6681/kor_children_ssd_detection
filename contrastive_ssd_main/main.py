import argparse
import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader
from myDataset import TripletDataset
from utils.collate import collate_fn
from utils.validation import validate, test
# from transformers import Wav2Vec2Processor
from train import train_model
from models.wav2vec2_encoder import Wav2Vec2Encoder
import debugpy
import random
import numpy as np

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_csv_path', type=str, required=True)
    parser.add_argument('--val_csv_path', type=str, required=True)
    parser.add_argument('--test_csv_path', type=str, required=True)
    
    parser.add_argument('--audio_column', type=str, required=True)
    parser.add_argument('--label_column', type=str, required=True)
    parser.add_argument('--text_column', type=str, required=True)
    
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model_path', type=str, default='contrastive_wav2vec2.pt')
    parser.add_argument('--train', action='store_true', help='flag when training')
    parser.add_argument('--run_inference', action='store_true', help='Run inference after training')
    # parser.add_argument('test_saved_checkpoint', type=str, default='Non'help='the path of saved checkpoint to test')
    return parser.parse_args()

def main():
    
    # debugpy.listen(5678)
    # print("waiting for debugger . . .")
    # debugpy.wait_for_client()
    
    seed = 42
    random.seed = seed
    np.random.seed = seed
    torch.manual_seed = seed
    torch.cuda.manual_seed_all = seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
        
    if args.wandb:
        wandb_run = wandb.init(project="ssd_contrastive",
                config={
                    # "encoder_model": args.encoder_model,
                    "train_datapath": args.train_csv_path,
                    "valid_datapath": args.val_csv_path,
                    "test_datapath": args.test_csv_path,
                    "hparam/batchsize" : args.batch_size,
                    "hparam/epochs" : args.epochs,
                    "hparam/learning_rate" : args.lr,
                })
        
    train_df = pd.read_csv(args.train_csv_path)
    # train_df = train_df.sample(n=1000) ##################################### DEBUG !!!
    train_dataset = TripletDataset(train_df, args.audio_column, args.label_column, args.text_column, seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn) # from the custom get_teim(), (anchor, pos, neg) * batch_size
    
    valid_df = pd.read_csv(args.val_csv_path)
    test_df = pd.read_csv(args.test_csv_path)
    
    if args.train:
    # main train loop
        best_thresholds, train_ref_embeddings = train_model(train_dataloader, train_df, valid_df, device, args) # thresholds leading to best uar obtained during validation
    
    if args.run_inference:
        
        # load best model
        model = Wav2Vec2Encoder().to(device)
        # if not args.test_saved_checkpoint is not None:
            # model.load_state_dict(torch.load(args.test_saved_checkpoint))
        # else:
        model.load_state_dict(torch.load(args.model_path))
        model.device = device
        
        test_acc, test_uar, classification_report = test(model, test_df, thresholds=best_thresholds, config=args)
        
        print(f"🧪 Test Accuracy: {test_acc:.4f}")
        print(f"🧪 Test UAR:     {test_uar:.4f}")
        print(classification_report)
        
if __name__ == "__main__":
        main()