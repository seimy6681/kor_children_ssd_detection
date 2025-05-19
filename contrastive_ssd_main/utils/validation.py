import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, recall_score, roc_curve, classification_report
from transformers import Wav2Vec2Processor
from models.wav2vec2_encoder import Wav2Vec2Encoder
import wandb

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def load_and_embed(model, audio_path):
    """
    loads audio from path and passes audio through model to obtain embedding
    """
    audio = torchaudio.load(audio_path)[0] # load audio from path
    
    # convert stero -> mono
    if audio.dim() == 2:
                audio = audio.mean(dim=0) # squeezed
    else:
        audio = audio.squeeze(0)
    
    # getting the embedding
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True)
    with torch.no_grad():
        z = model(inputs.input_values.to(model.device), attention_mask=inputs.attention_mask.to(model.device))
    return z.squeeze(0)


def compute_reference_embeddings(model, df, config):
    """
    computes average embeddings per target_text
    """
    ref_embeddings_per_text = {} # holds all the averaged correct embeddings for each target text
    grouped = df[df[config.label_column] == 0].groupby(config.text_column)
    for text, group in grouped:
        embeddings = []
        for audio_path in group[config.audio_column]:
            
            # audio = torchaudio.load(path)[0]
            # if audio.dim() == 2:
            #     audio = audio.mean(dim=0) # squeezed
            # else:
            #     audio = audio.squeeze(0)
            
            # # getting the embedding
            # inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            # with torch.no_grad():
            #     z = model(inputs.input_values.to(model.device), attention_mask=inputs.attention_mask.to(model.device))
            # z = z.squeeze(0)
            
            # above wrapped in embed()
            z = load_and_embed(model, audio_path)
            embeddings.append(z)  
            
        # stack embeddings get average embedding for current target text:
        if embeddings:
            ref_embeddings_per_text[text] = torch.stack(embeddings).mean(dim=0)
    return ref_embeddings_per_text
        
            
def compute_thresholds(model, df, ref_embeddings, config):
    """
    compute optimal threshold for each target text in the validation set
    iterates through the validation dataframe and compare each extracted audio embedding to its corresponding reference_embeddings 
    and compute the distances between the embeddings and the reference_embeddings and run roc_curve to 
    obtain optimal thresholds
    """
    
    distance_dict = {} # stores the distance between the reference embeddings obtained from current df and each embedding in the current df
    label_dict = {} # stores all label in the current df
    
    for i in range(len(df)):
        row = df.iloc[i]
        audio_path = row[config.audio_column]
        label = row[config.label_column]
        target_text = row[config.text_column]

        if target_text not in ref_embeddings:
            continue
        
        
        z = load_and_embed(model, audio_path)
        ref = ref_embeddings[target_text]
        distance = F.pairwise_distance(z.unsqueeze(0), ref.unsqueeze(0)).item()
        
        distance_dict.setdefault(target_text, []).append(distance)
        label_dict.setdefault(target_text, []).append(label)
    
    per_word_thresholds = {} # dictionary to store the optimal threshold for each target_text
    for text in distance_dict:
        distances = distance_dict[text] # list of euclidean distances between each sample and that word's reference embedding
        labels = label_dict[text] # list of corresponding binary labels
        
        # similarities = [-d for d in distances] # smaller distance -> higher similarity
        similarities = distances
        # convert inputs to rod_curve to cpu numpy arrays from lists
        labels = torch.tensor(labels).cpu().numpy()
        similarities = torch.tensor(similarities).cpu().numpy()
        
        fpr, tpr, thresholds = roc_curve(labels, similarities) # false positive rates, true positive rates, similarity thresholds that generate those (fpr, tpr) pairs
        youden_index = tpr - fpr
        best_idx = youden_index.argmax()
        best_threshold = thresholds[best_idx] # selecting optimal threshold for this word
        
        per_word_thresholds[text] = best_threshold
        print(f"[{text}] Optimal threshold τ: {best_threshold:.4f}")
    
    return per_word_thresholds
     
def predict(model, df, ref_embeddings, thresholds, config):
    y_true = []
    y_pred = []
    
    for i in range(len(df)):
        
        row = df.iloc[i]
        audio_path = row[config.audio_column]
        label = row[config.label_column]
        target_text = row[config.text_column]
        
        if target_text not in ref_embeddings or target_text not in thresholds:
            continue
        
        z = load_and_embed(model, audio_path)
        ref = ref_embeddings[target_text]
        threshold = thresholds[target_text]
        
        distance = F.pairwise_distance(z.unsqueeze(0), ref.unsqueeze(0))
        prediction = 0 if distance < threshold else 1
        
        y_true.append(label)
        y_pred.append(prediction)
        
    
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    
    return acc, uar, y_true, y_pred


def validate(model, train_df, df, config):
    
    model.eval()

    ref_embeddings = compute_reference_embeddings(model, train_df, config)
    thresholds = compute_thresholds(model, df, ref_embeddings, config)
    acc, uar, _, _ = predict(model, df, ref_embeddings, thresholds, config)
    
    return acc, uar, thresholds, ref_embeddings

def test(model, df, train_ref_embeddings, thresholds_from_val, config):
    """
    """
    model.eval()
    # compute ref_embeddings for target texts in the test set
    ref_embeddings = compute_reference_embeddings(model, df, config)
    # using test ref_embeddings and optima
    acc, uar, y_true, y_pred = predict(model, df, train_ref_embeddings, thresholds_from_val, config)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return acc, uar, report
    