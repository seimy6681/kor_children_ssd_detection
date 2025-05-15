import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor
import pandas as pd
import numpy as np
# from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import os
import math
import random
import jiwer
from util import Parser
import argparse, textwrap
from argparse import Namespace
import wandb
from typing import Tuple, Union
import debugpy
from sklearn.metrics import recall_score
import hangul_jamo
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
from train_asr import train_asr
from multitask_util.error_labels import generate_error_labels, gen_binary_label, gen_target_text_label, pad_to_max_length
from multitask_util.phoneme_error_labels import phoneme_error_label
from features.target_text_label import target_text_label
from transformers import Wav2Vec2Config

# torch.cuda.empty_cache()
# torch.cuda.set_per_process_memory_fraction()

# print(f"Python version: {sys.version}")
# import importlib.metadata
# print(importlib.metadata.version("Transformers version: transformers"))
# print(f"Transformers version: {transformers}")

print(f"Torch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

DATA_PATH="/data/selinawisco/kochild" # 음성파일과 CSV 파일 경로

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # DEBUGGING ------------------------------------------------------------
# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()

# ----------------------------------------------------------------------

#============================================
# arguments parsing
#============================================
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
default_group = parser.add_argument_group('default')
default_group.add_argument('--train', default=True, help='train mode')
default_group.add_argument('--num_runs', type=int, default=1, help='number of consecutive runs with consecutive random seeds')
default_group.add_argument('--eval_mode', default=False, action="store_true", help='model test mode')
default_group.add_argument('--checkpoint', type=str, help='model checkpoint to test')
default_group.add_argument('--data_path', type=str, default=DATA_PATH, help="directory path of data files")
default_group.add_argument('--task', type=str, default='asr_finetuning')
default_group.add_argument('--seed', type=int, default=42, help="random seed")
default_group.add_argument('--test_best_model', action="store_true")
default_group.add_argument('--output_dir', type=str)
default_group.add_argument('--result_csv', action="store_true")
default_group.add_argument('--debug', action="store_true")
default_group.add_argument('--reverse_alpha',type=float, help="how much to reverse the domain feature")
# wandb
wandb_group = parser.add_argument_group('wandb')
wandb_group.add_argument('--run_name', type=str, default="test", help="wandb run name")
wandb_group.add_argument('--logging_steps', type=int, default=50, help='wandb log & watch steps')
wandb_group.add_argument('--watch', type=str, default='all', help="wandb.watch parameter log")

# train args
train_group = parser.add_argument_group('train')
train_group.add_argument('--target', default='human_text_jamo', type=str, help="name of the target(column name in csv) for the custom loss function")
train_group.add_argument('--loss_feature', default=None, type=str, help="name of the feature(column name in csv) for the custom loss function")
train_group.add_argument('--multitask_alpha', type=float, help="weight for the multitask loss")
train_group.add_argument('--attention', type=str, help="check if using attended_binary model")
train_group.add_argument('--asr_mode',type=str, default='human', help='human : transcribing natural pronunciation, target : transcribing target text')
train_group.add_argument('--dropout',type=str, help='ex. --dropout = "0.1:6 7 8" -> dropout_rate=0.1, layers_to_apply = [6,7,8]')
train_group.add_argument('--adversarial',action='store_true')
train_group.add_argument('--embedding',action='store_true')
train_group.add_argument('--age_loss_weighting',action='store_true')

train_group.add_argument('--k_fold', type=int, help="k for k fold cross validation")
# train_group.add_argument('--callback', action='extend', nargs='*', type=str, 
#                          help=textwrap.dedent("""\
#                              callback list to use during training
#                              - es: early stopping
#                              - best: saving best model (watching vlidation loss)"""))
train_group.add_argument('--batch_size', type=int, default=8, help='batch size of training')
train_group.add_argument('--epochs', type=int, default=30, help='epochs of training')


# file 
file_group = parser.add_argument_group('file')
file_group.add_argument('--data_filename', type=str, default='hospital_target_all.csv')
file_group.add_argument('--splitted_data_files', default=False, action='store_true')
file_group.add_argument('--filter_dataset', default=False, action='store_true')
file_group.add_argument('--train_filename', type=str, help="file name of training data")
file_group.add_argument('--valid_filename', type=str, help="file name of validation data")
file_group.add_argument('--test_filename', type=str, help="file name of test data")



if __name__=='__main__':
    config = parser.parse_args()
    arg_parser = Parser()
    args = arg_parser(parser, config)
    
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    
    # Load the datasets ----------------------------------------------------------------------
    from datasets import load_dataset, Audio
    
    train_file = os.path.join(DATA_PATH, config.train_filename)
    test_file  = os.path.join(DATA_PATH, config.test_filename)
    
    train_dataset = load_dataset("csv", data_files={"train": train_file}, delimiter=",")["train"]
    test = load_dataset("csv", data_files={"test": test_file}, delimiter=",")["test"]
    

    ### 
    # train_dataset = train_dataset.filter(lambda row: row['age']>5)
    # test =test.filter(lambda row: row['age']>5)
    
    #remove unnecessary columns
    # train_dataset = train_dataset.remove_columns(['disease_type', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label' ])
    # test = test.remove_columns(['disease_type', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label'])
    train_dataset = train_dataset.rename_column("age", "age_label")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    test = test.cast_column("audio", Audio(sampling_rate=16_000))
    test = test.rename_column("age", "age_label")

    # Load Tokenizer, Feature Extractor, and Processor ----------------------------------------
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token='[UNK]', pad_token = "[PAD]", word_delimeter_token="|")
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)

    print(tokenizer.get_vocab())
    print(len(tokenizer.get_vocab()))


    from transformers import Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    
    # Preprocess audio and label data ---------------------------------------------------------

    def prepare_dataset(batch):
        
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        
        if config.loss_feature == 'target_text_id': 

            loss_feature = batch[config.loss_feature]  # List of cer_word entries corresponding to each audio
            orig_input = batch['input_values']
            new_input = np.append(orig_input, loss_feature).tolist() # append it at the end
            batch['input_values'] = new_input

        if config.loss_feature == 'error_label':
            
            # sparse label
            # if config.loss_feature == 'sparse_label':
            # loss_feature = np.array(generate_error_indices(batch['target_text_jamo'], batch['human_text_jamo']))
            
            # one hot encoding 
            loss_feature = np.array(generate_error_labels(batch['target_text_jamo'], batch['human_text_jamo']))
            
            orig_input = batch['input_values']
            new_input = np.append(orig_input, loss_feature).tolist() # append it at the end
            batch['input_values'] = new_input

        if config.loss_feature == 'binary_label':
            # print('binary_label')
            loss_feature = gen_binary_label(batch['target_text_jamo'],batch['human_text_jamo'])
            orig_input = batch['input_values']
            new_input = np.append(orig_input, loss_feature).tolist()
            batch['input_values'] = new_input

        if config.loss_feature == 'dual_label':
            text_loss_feature = gen_target_text_label(target_text_label.get(hangul_jamo.compose(batch['target_text_jamo'])))
            binary_loss_feature = gen_binary_label(batch['target_text_jamo'],batch['human_text_jamo'])
            orig_input = batch['input_values']
            new_input = np.append(orig_input, text_loss_feature) # +80
            new_input = np.append(new_input, binary_loss_feature) # +2
            batch['input_values'] = new_input
        
        if config.loss_feature == 'phoneme_error_label':
            phoneme_error_feature = phoneme_error_label(batch['target_text_jamo'], batch['human_text_jamo'])
            orig_input = batch['input_values']
            new_input = np.append(orig_input, phoneme_error_feature)
            batch['input_values'] = new_input
        
        if config.loss_feature == 'binary_classifier_by_phoneme':
            orig_input = batch['input_values']
            with processor.as_target_processor():
                target_phonemes = processor(batch['target_text_jamo']).input_ids
            target_phonemes = pad_to_max_length(target_phonemes)
            
            phoneme_error_feature = phoneme_error_label(batch['target_text_jamo'], batch['human_text_jamo'])
            new_input = np.append(orig_input, phoneme_error_feature) # +11
            new_input = np.append(new_input, target_phonemes) # +11
            
            batch['input_values'] = new_input
        
        if config.loss_feature == 'age_label':

            orig_input = batch['input_values']
            loss_feature = batch[config.loss_feature]
            new_input = np.append(orig_input, loss_feature)
            if config.age_loss_weighting:
                binary_loss_feature = np.argmax(gen_binary_label(batch['target_text_jamo'],batch['human_text_jamo']))
                new_input = np.append(new_input, binary_loss_feature)
            batch['input_values'] = new_input
        
        # Extract labels (transcripts) as usual
        with processor.as_target_processor():
            batch['labels'] = processor(batch[config.target]).input_ids
        return batch 

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,num_proc=1)
    test_dataset = test.map(prepare_dataset, remove_columns=test.column_names,num_proc=1)
    
    MIN_LENGTH =320 * 10 # downsampling factor , mask_time_length
    train_dataset = train_dataset.filter(lambda x: len(x['input_values']) >= MIN_LENGTH)
    
    # Define the DATA COLLATOR with loss feature processing  -------------------------------------
    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
        
        processor: Wav2Vec2Processor
        padding: Union[bool,str]= True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            
                
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            
            if config.loss_feature  == 'target_text_id':
                loss_feature_batch = torch.tensor([feature['input_values'][-1] for feature in input_features], dtype=torch.float32)
            
                # remove loss feature from input values
                for feature in features:
                    feature['input_values'] = feature['input_values'][:-1]
                    
            elif config.loss_feature  == 'error_label':
                loss_feature_batch = torch.tensor([feature['input_values'][-57:] for feature in input_features], dtype=torch.float32)

                # remove error_vector(57) from input values
                for feature in features:
                    feature['input_values'] = feature['input_values'][:-57]
            
            elif config.loss_feature == 'binary_label':
                loss_feature_batch = torch.tensor([feature['input_values'][-2:] for feature in input_features], dtype=torch.float32)

                for feature in features:
                    feature['input_values'] = feature['input_values'][:-2]
                    
            elif config.loss_feature == 'phoneme_error_label':
                loss_feature_batch = torch.tensor([feature['input_values'][-11:] for feature in input_features], dtype=torch.float32)
            
                for feature in features:
                    feature['input_values'] = feature['input_values'][:-11]
                    
            elif config.loss_feature == 'dual_label':
                # target_text_label (80) + binary_label (2)
                binary_feature_batch = torch.tensor([feature['input_values'][-2:] for feature in input_features], dtype=torch.float32)

                for feature in features:
                    feature['input_values'] = feature['input_values'][:-2]

                # update input_features
                input_features = [{"input_values": feature["input_values"]} for feature in features]

                text_feature_batch = torch.tensor([feature['input_values'][-80:] for feature in input_features], dtype=torch.float32)

                for feature in features:
                    feature['input_values'] = feature['input_values'][:-80]
                    
            elif config.loss_feature == 'binary_classifier_by_phoneme':
                # target_text_label (80) + binary_label (2)
                target_phonemes_batch = torch.tensor([feature['input_values'][-11:] for feature in input_features], dtype=torch.float32)

                for feature in features:
                    feature['input_values'] = feature['input_values'][:-11]

                # update input_features
                input_features = [{"input_values": feature["input_values"]} for feature in features]

                loss_feature_batch = torch.tensor([feature['input_values'][-11:] for feature in input_features], dtype=torch.float32)

                for feature in features:
                    feature['input_values'] = feature['input_values'][:-11]

            elif config.loss_feature == 'age_label':
                if config.age_loss_weighting:
                    # grabbing binary label
                    binary_feature_batch = torch.tensor([feature['input_values'][-1:] for feature in input_features], dtype=torch.float32)
                    
                    #getting rid of the last value in each input_values (the binary label)
                    for feature in features:
                        feature['input_values'] = feature['input_values'][:-1]

                    # updating input_features with the updated feature without binary_label at the end
                    input_features = [{"input_values": feature["input_values"]} for feature in features]

                    #grabbing age now
                loss_feature_batch = torch.tensor([feature['input_values'][-1:] for feature in input_features], dtype=torch.float32) 

                for feature in features: # modifying the actual feature passed in as parameter so age is omitted from audio value
                    feature['input_values'] = feature['input_values'][:-1]
                
            batch = self.processor.pad(
                input_features,
                padding = self.padding,
                max_length = self.max_length,
                pad_to_multiple_of = self.pad_to_multiple_of,
                return_tensors="pt",
            )
            
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding = self.padding,
                    max_length = self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            
            if config.loss_feature == 'dual_label':
                batch['target_text_label'] = text_feature_batch
                batch['binary_label'] = binary_feature_batch

            elif config.loss_feature == 'binary_classifier_by_phoneme':
                batch['target_phonemes'] = target_phonemes_batch
                batch['phoneme_error_label'] = loss_feature_batch

            elif config.age_loss_weighting:
                batch[config.loss_feature] = loss_feature_batch
                batch['binary_label'] = binary_feature_batch
                 
            elif config.loss_feature != 'phoneme' and config.loss_feature != 'baseline':
                batch[config.loss_feature] = loss_feature_batch
            
            # for i in range(len(batch['input_values'])):
                
            #     print(batch['input_values'][i].shape)
            return batch
        
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Define the metrics-----------------------------------------------------------------------
    
    def compute_metrics(pred):
    # Accessing the ASR logits
        if isinstance(pred.label_ids, tuple):  # In multitask case, label_ids is be a tuple
            asr_labels = pred.label_ids[0]  # The first part is for ASR task (CTC labels)
            multitask_labels = pred.label_ids[1]  # The second part is for binary classification
        else:
            asr_labels = pred.label_ids
        
        if isinstance(pred.predictions, tuple):
            pred_logits = pred.predictions[0]
            multitask_logits = pred.predictions[1]  # Binary task predictions
        else:
            pred_logits = pred.predictions  # ASR logits (shape: batch_size, seq_len, num_classes)
            multitask_logits = None

        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Decode the ASR predictions and compute CER
        asr_labels[asr_labels == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(asr_labels, skip_special_tokens=True)  # Skip special tokens in labels
        # pred_str = processor.batch_decode(pred_ids, group_tokens=False)  # Skip special tokens in predictions
        # label_str = processor.batch_decode(asr_labels, group_tokens=False)  # Skip special tokens in labels

        # Calculate CER (Character Error Rate)
        cer = jiwer.cer(pred_str, label_str)
        
        # for i in range(3):
        #     print(f'Prediction: {pred_str[i]}')
        #     print(f"Label: {label_str[i]}")

        # If binary_logits are present (binary classification task)
        if multitask_logits is not None and config.loss_feature == 'binary_label':
            
            binary_preds = np.argmax(multitask_logits, axis=-1)  # Get binary predictions
            binary_labels = np.argmax(multitask_labels, axis=-1)  # Get binary labels (assuming one-hot encoded)
            binary_accuracy = np.mean(binary_preds == binary_labels)  # Calculate binary accuracy
            binary_uar = recall_score(binary_labels, binary_preds, average='macro')
            
            return {"cer": cer, "binary_UAR" : binary_uar, "binary_accuracy": binary_accuracy}
        if multitask_logits is not None and config.loss_feature == 'phoneme_error_label' or config.loss_feature == 'binary_classifier_by_phoneme':
            binary_preds = np.argmax(multitask_logits, axis=-1)
            
            # Mask valid indices (exclude -100)
            valid_mask = multitask_labels != -100  # Boolean mask for valid labels

            # Filter out invalid values
            binary_preds = binary_preds[valid_mask]
            binary_labels= multitask_labels[valid_mask]
            
            binary_accuracy = np.mean(binary_preds == binary_labels)
            binary_uar = recall_score(binary_labels, binary_preds, average='macro')
            
            return {"cer": cer, "phoneme_error_classification_UAR" : binary_uar, "phoneme_error_classification_accuracy": binary_accuracy}
        
        if multitask_logits is not None and config.loss_feature == 'age_label':
            
            age_preds = np.argmax(multitask_logits, axis=-1)+2 # get age classification
            age_labels = multitask_labels.squeeze(1) # use it straight away without argmax
            age_accuracy = (age_preds == age_labels).sum() / age_labels.shape[0] # accuracy_score(age_labels,age_preds) from sklearn.metrics
            age_uar = recall_score(age_labels, age_preds, average='macro')
        
            return {"cer": cer, "age_UAR": age_uar, "age_accuracy": age_accuracy}

        return {"cer": cer}
    
    # defining config
    hf_config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        attention_dropout = 0.0,
        hidden_dropout = 0.0,
        mask_time_prob=0.05,
        mask_time_length = 5,
        layerdrop = 0.0,
        ctc_loss_reduction = "mean",
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    if config.multitask_alpha:
        hf_config.multitask_alpha = config.multitask_alpha

    # Load the model -----------------------------------------------------------------------
    from transformers import Wav2Vec2ForCTC

    for seed in range(config.seed, config.seed + config.num_runs): 
        print(f"Running training with seed {seed}")

        # Set the new seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU setups


        if config.loss_feature == 'error_label':
            from models.multitask_wav2vec2 import MultitaskWav2Vec2ForCTC

            model = MultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config,main_arg=config)

        elif config.loss_feature == 'binary_label':
            if config.attention == 'single': 
                from models.attended_binary_multitask_wav2vec2 import AttendedBinaryMultitaskWav2Vec2ForCTC
                print('using second attention layer!')
                model = AttendedBinaryMultitaskWav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config)
            elif config.attention == 'dynamic': 
                from models.dynamic_attended_binary_multitask_wav2vec2 import DynamicAttendedBinaryMultitaskWav2Vec2ForCTC
                print('using dynamic multitask + attention layer!')
                model = DynamicAttendedBinaryMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config
            )
            elif config.attention == 'joint': 
                from models.joint_attended_multitask_wav2vec2 import JointAttendedBinaryMultitaskWav2Vec2ForCTC
                print('using joint multitask + attention layer!')
                model = JointAttendedBinaryMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config
            )
            elif config.attention == 'multihead':
                from models.multihead_binary_multitask_wav2vec2 import MultiheadAttentionMultitaskWav2Vec2ForCTC
                print('using multihead second attention layer!')
                model = MultiheadAttentionMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config
                )
            else:
                from models.binary_multitask_wav2vec2 import BinaryMultitaskWav2Vec2ForCTC
                model = BinaryMultitaskWav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config
                    )
        elif config.loss_feature == 'dual_label':
            from models.dual_multitask_wav2vec2 import DualMultitaskWav2Vec2ForCTC
            model = DualMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config

            )        
        elif config.loss_feature == 'phoneme_error_label':
            from models.phoneme_binary_multitask_wav2vec2 import PhonemeBinaryMultitaskWav2Vec2ForCTC
            model = PhonemeBinaryMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config)
            
        elif config.loss_feature == 'binary_classifier_by_phoneme':
            from models.by_phoneme_classifier_multitask_wav2vec2 import ByPhonemeClassifierMultitaskWav2Vec2ForCTC
            print('by phoneme binary classifier model running!')
            model = ByPhonemeClassifierMultitaskWav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config
            )
            

        ## Age 이용
        elif config.loss_feature == 'age_label':
            if config.adversarial:
                from models.dann_age_classifier_wav2vec2 import AgeAdversarialWav2Vec2ForCTC
                print("training Wav2Vec2CTC + domain adversarial learning with age!")
                model = AgeAdversarialWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config)
            elif config.embedding:
                from models.age_embedding_wav2vec2 import AgeEmbeddingWav2Vec2ForCTC
                model = AgeEmbeddingWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config)
            elif config.age_loss_weighting:
                print('weighting loss 5 times when')
                from models.age_loss_weighting_wav2vec2 import AgeLossWeightingWav2Vec2ForCTC
                model = AgeLossWeightingWav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xls-r-300m',config=hf_config,main_arg=config)
            else:  
                from models.age_classifier_multitask_wav2vec2 import AgeClassifierWav2Vec2ForCTC
                print("training Wav2Vec2CTC + age classification!")
                model = AgeClassifierWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", config=hf_config, main_arg=config)
        ## Baseline 
        elif config.loss_feature == None:
            print("training baseline Wav2Vec2CTC!")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", config=hf_config)
            print(model.config.mask_time_length)

        model.config.alpha = config.multitask_alpha
        model.to(device)
        
        # Wrap the model with DataParallel for parallel processing when multi GPU
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # if config.dropout != None :
        #     dropout_input = config.dropout.split(":")
        #     dropout_rate = float(dropout_input[0])
        #     dropout_layers = list(map(int, dropout_input[1].split(' '))) #  ex. --dropout = "0.1:6 7 8" -> dropout_rate=0.1, layers_to_apply = [6,7,8]
            
        model.freeze_feature_encoder()
        model.gradient_checkpointing_enable()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
        
        # train! 
        if not config.eval_mode:
            train_asr(model, data_collator, processor, tokenizer,feature_extractor, train_dataset, test_dataset, compute_metrics, config, seed)