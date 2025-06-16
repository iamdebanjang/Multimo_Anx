# Multimodal Generative Anxiety & Stress Detector  


---

## 🌟 Why this project?

Stress and anxiety affect health, productivity, and safety, yet most monitoring tools rely on questionnaires or single-sensor signals.  
Our approach fuses multiple human signals—**voice, video, text, and optional physiology**—and adds four new ideas that make the system more robust and personalised:

1. **Diffusion Imputation (CDI)** – recreates an entire missing modality on-the-fly.  
2. **Uncertainty-Aware Fusion (UAF)** – down-weights noisy or low-confidence inputs automatically.  
3. **Self-Supervised Cross-Modal Pre-training (SCP)** – learns generic audio–video–text links before seeing a single stress label.  
4. **Subject-Level Bayesian Adaptation (SBA)** – fine-tunes the last layer analytically when a new user provides a handful of calibration samples.

> **Novelty claim**: no prior stress/anxiety pipeline combines *all four* ideas in one framework.

---

## 🗂 Directory layout

```
.
├─ anxiety_stress_multimodal.py   # model that runs out-of-box
├─ data/                          # put processed tensors here (Coming soon)
│   ├─ daic_woz/
│   ├─ wesad/
│   └─ ...
├─ notebooks/                     # analysis / visualisations 
├─ docs/                          # architecture diagrams, banner.png
├─ requirements.txt
└─ README.md                      
```



## 🧩 Model components 

| Block | What it does | Where to swap in a stronger one |
|-------|--------------|---------------------------------|
| **SimpleEncoder** | Turns a signal sequence into a 256-D embedding + an uncertainty score | Replace with wav2vec 2.0, ViT, BERT, or any backbone |
| **UAF** | Combines embeddings, giving less weight to noisy ones | Keep—works out-of-the-box |
| **VAE bottleneck** | Creates a latent “emotion fingerprint” and supports reconstruction loss | Swap for InfoVAE or disable with `use_vae=False` |
| **DiffusionImputerStub** | Generates zeros when a modality is missing | Plug in a conditional diffusion model from 🤗 *diffusers* |
| **BayesianHead** | Personalises predictions with a closed-form update | Keep; prior variance is tunable |

---

## 🔬 Datasets supported

| Dataset | Modalities | What you detect |
|---------|------------|-----------------|
| **DAIC-WOZ**          | Audio, Video, Text | Depression / Anxiety |
| **WESAD**             | Video, Physio      | Stress vs. neutral   |
| **MuSE (Emotion)**    | Audio, Video, Text | Continuous valence / arousal |
| **ForDigitStress**    | Video, Physio      | Cognitive stress     |

See `docs/preprocess.md` for how to turn each raw dataset into tensors.

---

## 🏃 Training on real data

1. Pre-extract features for each modality (MFCCs, 2D landmarks, tokens, ECG windows).  
2. Save as tensors:
   ```
   data/daic_woz/train_audio.pt
   data/daic_woz/train_visual.pt
   ...
   ```
3. Edit `anxiety_stress_multimodal.py` to point encoders at your backbone checkpoints.  
4. Replace the toy `for step in range(5)` loop with a DataLoader and real epochs.  
5. Tune the loss mix in  
   ```python
   loss = clf_loss + 0.1 * recon_loss + 0.01 * kld_loss
   ```

---

## 📊 Evaluation script (coming soon)

```bash
python eval.py --ckpt best.pt --dataset daic_woz --metrics f1 auc
```

Metrics we report:

- **Classification** – Accuracy, Precision, Recall, F1, AUC-ROC  
- **Regression** – RMSE, MAE, R²  
- **Generative quality** – Reconstruction MSE, FID (audio/image)

---

## 🆚 Baselines we compare against

| Baseline | Modalities | Original paper |
|----------|------------|----------------|
| StressNet | Physio + Audio | He et al. 2022 – [arXiv:2206.04270] |
| MuSe Transformer | Audio + Video + Text | Stappen et al. 2021 – ACM MM |
| AVEC CNN-LSTM | Audio + Video | Valstar et al. 2016 – AVEC |
| DAIC-WOZ Transformer | Audio + Video + Text | Mallol-Ragolta et al. 2019 |
| WESAD CNN-LSTM | Physiological | Schmidt et al. 2018 – IMWUT |

---

## 🛡 Ethics & privacy

* All personal data must be anonymised and encrypted at rest.  
* Users may opt-out of specific sensors at any time.  
* Attention heat-maps and uncertainty scores provide transparency, helping clinicians trust the output.

---

## 🛠 Roadmap

- [ ] Swap the diffusion imputer stub for a real conditional UNet  
- [ ] Add official DataLoader scripts for DAIC-WOZ, WESAD, MuSE  
- [ ] Integrate Weights & Biases training logger  
- [ ] Publish full reproduction notebooks with ablation plots


