# ASR-SSD Main: Korean Children Speech Sound Disorder Detection using ASR

This directory contains an Automatic Speech Recognition (ASR) framework built on top of **Wav2Vec2-CTC** to detect mispronunciation and Speech Sound Disorders (SSD) in Korean children's speech. The codebase extends Hugging Face fine-tuning scripts to support modular multi-task learning, attention mechanisms, and age-aware loss formulations.

---

## 🛠️ Required Code Modifications Before Running

Before launching training or inference, update the dataset path in `main.py`:

* Open `main.py` and modify the `DATA_PATH` variable near the top of the file:
  ```python
  DATA_PATH = '/path/to/your/csv_and_audio_files_directory'