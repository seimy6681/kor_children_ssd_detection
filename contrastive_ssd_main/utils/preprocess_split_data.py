from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv("/home/selinawisco/sel-hub/asr/r08.1_train.csv")

# First stratify split into train+valid and test
train_valid, test = train_test_split(df, test_size=0.15, stratify=df[["target_text", "new_label"]], random_state=42)

# Then stratify split train+valid into train and valid
train, valid = train_test_split(train_valid, test_size=0.1765, stratify=train_valid[["target_text", "new_label"]], random_state=42)
# Why 0.1765? Because 15% of remaining 85% ≈ 15% total

# Optional: reset indices
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_csv('/home/selinawisco/ssd_contrastive/datasets/r1.train.csv', index=False)
valid.to_csv('/home/selinawisco/ssd_contrastive/datasets/r1.valid.csv', index=False)
test.to_csv('/home/selinawisco/ssd_contrastive/datasets/r1.test.csv', index=False)
