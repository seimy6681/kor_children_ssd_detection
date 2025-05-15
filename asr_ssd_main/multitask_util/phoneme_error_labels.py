import torch
import pandas as pd 

# finding the length of the longest target_text in the training set r08.1_train
# df = pd.read_csv('/home/selinawisco/sel-hub/asr/r08.1_train.csv')
# max_len = df['target_text_jamo'].astype(str).str.len().max() # 11

def phoneme_error_label(target, actual):
    err_label = torch.full((11,), -100, dtype=torch.float32) 

    for idx, (t, a) in enumerate(zip(target, actual)):
              
        err_label[idx] = 0 if (t == a) else 1

    return err_label