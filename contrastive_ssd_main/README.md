### Contrastive Learning for Korean Children SSD Detection

## Model Architecture
- **Triplet Loss Objective:** This model employs `TripletLoss` to learn discriminative speech representations:
  - **Anchor:** A randomly selected speech sample.
  - **Positive Pair:** A sample containing the **same target word** and the **same pronunciation label** (correct/incorrect) as the anchor.
  - **Negative Pair:** A sample containing the **same target word** but with the **opposite pronunciation label**.
  - **Objective:** Minimize the embedding distance between positive pairs while maximizing the distance between negative pairs in latent space.
- **Implementation:** Implemented in PyTorch, leveraging contextualized hidden state features extracted from a Wav2Vec2 Encoder.

---

## Training

- **Execution:**
  ```bash
  ./run.sh