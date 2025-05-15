import numpy as np
import torch
import json
import torch.nn.functional as F

def align_sequences(target, actual):
    """
    Align the target and actual sequences by inserting '-' where addition or omission occurs.
    
    Args:
        target (list): The target sequence of characters.
        actual (list): The actual sequence of characters.
    
    Returns:
        aligned_target (list): The aligned target sequence with '-' for additions.
        aligned_actual (list): The aligned actual sequence with '-' for omissions.
    """
    
    aligned_target = []
    aligned_actual = []
    i, j = 0,0
    
    while i < len(target) and j < len(actual):
        if target[i] == actual[j]:
            aligned_target.append(target[i])
            aligned_actual.append(actual[j])
            # print(f'pass! {target[i]} {actual[j]}')
            i += 1
            j += 1
            
        elif i + 1 < len(target) and target[i+1] == actual[j]:
            # Omission: next target char matches curr actual char
            aligned_target.append(target[i])
            aligned_actual.append('-')
            print(f'omission! {target[i]} {actual[j]}')
            i += 1
            
        elif j+1 < len(actual) and actual[j+1] == target[i]:
            # Addition: next actual character matches curr target char
            aligned_target.append('-')
            aligned_actual.append(actual[j])
            print(f'addition! {target[i]} {actual[j]}')
            j+=1
            
        else:
            # substitution
            aligned_target.append(target[i])
            aligned_actual.append(actual[j])
            print(f'substitution! {target[i]} {actual[j]}')
            i += 1
            j += 1
            
            
    while i < len(target):
        aligned_target.append(target[i])
        aligned_actual.append('-')
                
        i += 1
    while j < len(actual):
        aligned_target.append('-')
        aligned_actual.append(actual[j])
        
        j += 1
            
    return aligned_target, aligned_actual
            
            
            

'''
Helper function to generate error label vectors for the auxilary error detection task in the multitask_wav2vec2 model

'''

def generate_error_labels(target,actual):
    """
    Generate a 57-dimensional binary error vector for consonant errors.
    """

    error_vector = [0] * 57
    consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ','ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ','ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㄲ', 'ㄸ', 'ㅃ','ㅆ','ㅉ']
    error_types = {'substitution': 0, 'omission': 1, 'addition': 2}

    # Align the target and actual sequences
    aligned_target, aligned_actual = align_sequences(target, actual)

    # Compare aligned sequences
    for t_char, a_char in zip(aligned_target, aligned_actual):
        if t_char in consonants and a_char in consonants and t_char != a_char:
            # Substitution error
            idx = consonants.index(t_char) * 3 + error_types['substitution']
            error_vector[idx] = 1
            # print(f'substitution detected at {t_char}  inserting 1 at {idx}')
        elif t_char in consonants and a_char == "-":
            # Omission error
            idx = consonants.index(t_char) * 3 + error_types['omission']
            error_vector[idx] = 1
            # print(f'omision detected at {t_char}  inserting 1 at {idx}')
            
        elif t_char == "-" and a_char in consonants:
            # Addition error
            idx = consonants.index(a_char) * 3 + error_types['addition']
            error_vector[idx] = 1
            # print(f'addition detected at {t_char}  inserting 1 at {idx}')
            
            
    return error_vector


def gen_binary_label(target,actual):
    if target==actual:
        return np.array([1, 0])
    else:
        return np.array([0, 1])
    

def gen_target_text_label(text_idx):
    '''
    input: the index of the current target text(0-79)
    output: 80 zeros array except a 1 at the text_idx 
    '''

    target_text_vector = [0] * 80
    target_text_vector[text_idx] = 1

    return target_text_vector

def phoneme_error_label(target, actual):
    err_label = torch.full((11,), -100, dtype=torch.float32) 

    for idx, (t, a) in enumerate(zip(target, actual)):
              
        err_label[idx] = 0 if (t == a) else 1

    return err_label

def pad_to_max_length(target):
    pad_len = 11 - len(target)
    if pad_len > 0 :
        target = F.pad(torch.tensor(target), (0,pad_len), value=0)
   
    return target
    




    