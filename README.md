# Movement Classification from Corticomotor Neuron Activity  
**Cross-Subject Generalization**

Daniel Katz & Roi Wayner  
Deep Learning for Physiological Signals (3360209)  

---

## 📖 Overview
This project implements and evaluates algorithms for **cross-subject movement classification** using corticomotor neuron recordings and EMG data from two Rhesus monkeys (Chip & Gabby). We compare simple baselines (logistic regression) with deep-learning approaches (1D-residual CNN and LFADS-latent-space + CNN) to classify 12 cued digit/wrist movements and assess generalization from one subject to another.

---

## 🧪 Experimental Design
- **Subjects:** 2 Rhesus monkeys (Chip, Gabby)  
- **Tasks:** 12 cued movements (flexion/extension of digits 1–5 and the wrist)  
- **Recordings:**  
  - Single CM-cell spike trains (Chip: 179 cells, Gabby: 119 cells)  
  - Forearm/hand EMG (Chip: 30 channels, Gabby: 22 channels)  
  - Kinematic labels via strain gauges & microswitches  

---

## 🔄 Data Processing Pipeline
1. **Preprocessing:**  
   - Align spike trains, EMG, and movement labels  
   - Normalize / bin neural and EMG signals  
2. **Dimensionality Reduction:**  
   - PCA on neural data for visualization  
3. **Cross-Subject Split:**  
   - Train on Chip → Test on Gabby  

---

## 🛠 Models  
- **Logistic Regression** — classical baseline  
- **1D-ResCNN** — residual convolutional network on binned spikes  
- **LFADS + CNN** — LFADS-inferred latent factors fed into a CNN classifier  

---

## 📈 Key Results  
- **1D-ResCNN** achieved an overall accuracy of **0.92** on cross-subject classification.  
- LFADS + CNN similarly provided strong performance and interpretable latent features.  
- **Interpretability:** Attention maps highlight top-contributing neurons, e.g. `C0137`, `C0027`, `C0115`, `c0491`, `C0235`.  

*(Full metrics and per-class F1 scores available in `notebooks/results_visualization.ipynb`.)*

---

## 📂 Repository Structure
```plaintext
cross-subject-movement-decoding/
├── data/
│   ├── raw/           # Original spike, EMG, kinematic files
│   └── processed/     # Aligned .npy/.csv for modeling
├── models/
│   ├── logistic_regression.py
│   ├── res_cnn.py
│   └── lfads_cnn.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── presentations/
│   └── final_daniel_roi.pdf
├── requirements.txt
└── README.md
```

🚀 Installation
```
git clone https://github.com/<your-username>/cross-subject-movement-decoding.git
cd cross-subject-movement-decoding
pip install -r requirements.txt
```
🧩 Quick Start
Preprocess raw data
```
python scripts/preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed
```

Train a model
```
python scripts/train.py \
  --model res_cnn \
  --data_dir data/processed \
  --epochs 50 \
  --batch_size 32
```

Evaluate cross-subject generalization

```
python scripts/evaluate.py \
  --model res_cnn \
  --train_subject Chip \
  --test_subject Gabby
  ```
📜 Citation

If you use this work, please cite:

Movement Classification From Corticomotor Neuron Activity: Cross Subject Generalization
Daniel Katz & Roi Wayner, Deep Learning for Physiological Signals, 2025.

📝 License
This project is released under the MIT License.
