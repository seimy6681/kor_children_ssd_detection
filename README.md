# Korean Children Speech Sound Disorder (SSD) Detection

This repository contains Automatic Speech Recognition (ASR) and audio classification frameworks for detecting mispronunciation and Speech Sound Disorders (SSD) in pediatric speech. Built on top of PyTorch and Hugging Face Transformers, it features custom model architectures, multi-task learning paradigms, and specialized data augmentation pipelines.

---

## 📁 Directory & Methodology Overview

### 1. `asr_ssd_main/`
Wav2Vec2-based ASR and multi-task learning architectures. The codebase extends Hugging Face fine-tuning scripts to allow modular configuration of model architectures and loss functions via command-line arguments.

* **Custom Models (`asr_ssd_main/models/`):**
  * **ASR + Age Classifier:** Multi-task learning model for joint speech recognition and age estimation.
  * **ASR + Age Embedding:** Conditioning model features on child age embeddings.
  * **ASR + Custom Age-Weighted Loss:** Loss formulation weighted by speaker age.
  * **ASR + Fused Attention:** Weakly supervised multi-task model using fused attention for binary classification *(Presented in our INTERSPEECH 2025 paper: "Multitask Learning with Fused Attention for Improved ASR and Mispronunciation Detection in Children's Speech Sound Disorders")*.
  * **ASR + Adversarial Learning:** Mitigating speaker age bias using adversarial training.

* **Execution & Usage:**
  * **Training:** Run `./run.sh` to trigger the selected setup via `train_asr.py`.
  * **Inference & Metrics:** Specify `MODEL_PATH` in `inference.py` to calculate Character Error Rate (CER) and Unweighted Average Recall (UAR).
  * *For detailed execution instructions, see [`asr_ssd_main/README.md`](./asr_ssd_main/README.md).*

---

### 2. `whisper_asr_huggingface/`
Fine-tuning scripts and custom multi-task architectures leveraging OpenAI's Whisper model.

* **Custom Models (`whisper_asr_huggingface/whisper_models/`):**
  * **ASR + Binary Classification:** Joint speech recognition and disorder detection.
  * **Fused Attention Multi-Task Model:** Multi-task architecture combining ASR and disorder detection *(INTERSPEECH 2025)*.
  * **ASR + Phoneme Classification:** Frame-level phoneme classification using forced alignment targets.
* **Forced Alignment Utilities (`whisper_asr_huggingface/util/`):**
  * CTC-based forced alignment utilities for audio-text alignment.

---

### 3. `augmentation/`
Speech data synthesis and augmentation pipelines tailored for pediatric acoustic datasets.

* **TTS Pipeline (`augmentation/tacotron2/`):** Tacotron 2 + WaveGlow Text-to-Speech (TTS) synthesis system.
* **Audio Mixing Utilities (`augmentation/mixing_and_others/`):**
  * `age_mixing.ipynb`: Age-based audio mixing for data augmentation *(Presented in our [INTERSPEECH 2024 paper](https://www.isca-archive.org/interspeech_2024/kim24q_interspeech.html))*.
  * `phoneme_mixing.ipynb`: Phoneme-level audio mixing utilizing CTC-based forced alignment.

---

### 4. `contrastive_ssd_main/`
PyTorch implementation of a contrastive learning framework designed for representation learning in audio classification.