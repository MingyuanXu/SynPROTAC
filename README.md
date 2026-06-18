# SynPROTAC

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-GPLv3-green)

**Design synthesizable PROTACs through synthesis constrained generative model and reinforcement learning.**

---

## 📖 Overview

Protein hydrolysis targeting chimeric (PROTAC) has emerged as a promising technology in degrading disease-related proteins for drug design. Recent deep generative models can accelerate PROTAC design, but the generated molecules are often difficult to synthesize. Here we develop **SynPROTAC** model, which integrates chemical reaction path driven molecule assembly with reinforcement learning for design of synthesizable PROTACs together with favorable binding properties. Specifically, the synthesis constrained generative model employs Graphormer encoded warhead or E3 ligand as input, and autoregressively samples reaction templates and building blocks through transformer based decoder and chemical fingerprint based searching along with transfer learning for PROTAC construction. The comprehensive evaluations indicated that SynPROTAC is capable of generating new PROTACs with feasible synthetic routes, reasonable physico-chemical and binding related properties. We further applied SynPROTAC to design PROTAC molecules degrading bromodomain-containing protein 4 (BRD4), and two selected compounds were successfully synthesized according to the synthetic routes proposed by SynPROTAC. In the following biological experiments, both of them exhibited nanomolar-level degradation activity against BRD4 and potent anti-proliferation activity against MV411 tumor cell.

### Key Features

- 🧬 **Synthesis-constrained generative model** — Graphormer-encoded warhead/E3 ligand with autoregressive transformer decoder for reaction template and building block sampling
- 🧪 **Reinforcement learning** — Optimize binding properties (docking scores, 2D similarity, and more)
- 🏗️ **MCTS-based retrosynthetic planning** — Multi-step synthetic route exploration

---

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MingyuanXu/SynPROTAC.git
cd SynPROTAC
```

### 2. Create conda environment

```bash
conda env create -f environment.yaml
conda activate synprotac
```

### 3. Install SynPROTAC package

```bash
pip install -e .
```

---

## 📦 Data & Pretrained Models

The repository includes a toy dataset for testing the training workflow. The full dataset (20 million synthesizable routes) and the pretrained model can be downloaded from:

- **Figshare**: [The pretrained model of SynPROTAC](https://figshare.com/articles/journal_contribution/The_pretrained_model_of_SynPROTAC_/30446639)

After downloading, place the checkpoint into the `pretrained_models/` directory:

```bash
mv synprotac_prior.ckpt pretrained_models/
```

### Data files

| File | Description |
|------|-------------|
| `data/reagents.txt` | Building block reagents library |
| `data/templates.txt` | Reaction templates library |
| `data/warhead_dealed.txt` | Warhead molecule library |
| `data/e3_ligand_dealed.txt` | E3 ligand molecule library |
| `data/testset.csv` | Test set |

---

## 🧪 Usage

### Training

Train the synthesis-constrained generative model:

```bash
cd scripts/train
python train.py -i ctrl.json
```

> **Note**: The provided `ctrl.json` uses a toy dataset. For full training, download the complete dataset from Figshare.

### Sampling / Molecule Generation

Generate new PROTAC molecules with synthetic routes:

```bash
cd scripts/sample
python sample.py -i ctrl.json
python show_path.py
```

### Evaluation

Evaluate generated PROTACs:

```bash
cd scripts/eval
python eval.py -i ctrl.json
```

---

### Reinforcement Learning

SynPROTAC supports multiple RL strategies for optimizing PROTAC properties.

#### 🔹 2D Similarity Guided RL

```bash
cd scripts/rl/2D_similarity
python rl.py -i ctrl.json
```

#### 🔹 Constrained Docking Guided RL

```bash
cd scripts/rl/Constrained_Docking
python rl.py -i ctrl.json
```

#### 🔹 RL without Crystal Structure 🆕

When target-crystal structures are unavailable, SynPROTAC can perform RL-based optimization using homology models or AlphaFold-predicted structures. Four example targets are provided:

| Target | Description | Model File |
|--------|-------------|------------|
| **CDK9** | Cyclin-dependent kinase 9 | `CDK9_model.pdb` |
| **VHL-FKBP5** | FKBP5 recruited by VHL | `8pc2_model.pdb` |
| **VHL-SMARCA2** | SMARCA2 recruited by VHL | `7277_model_D1416_localfix.pdb` |
| **WDR5-VHL1** | WDR5 targeted with VHL E3 ligase | `7q2j_model.pdb` |

Run an example (e.g., WDR5-VHL1):

```bash
cd scripts/rl_without_Crystal_struct/WDR5-VHL1
python train.py -i ctrl.json
```

**Configuration** (`ctrl.json`):
```json
{
    "CUDA_VISIBLE_DEVICES": "0",
    "batchsize": 20,
    "learning_rate": 1e-5
}
```

Each example directory contains:
| File | Description |
|------|-------------|
| `*.pdb` | Protein model structure (homology model with Schordinger) |
| `ref_ligand.sdf` | Reference ligand for binding site definition |
| `reagents.txt` | Building block library |
| `templates.txt` | Reaction template library |
| `ctrl.json` | Training hyperparameters |
| `train.py` | RL training script |

---

## 📁 Project Structure

```
SynPROTAC/
├── data/                              # Dataset files
│   ├── reagents.txt                   # Building block reagents
│   ├── templates.txt                  # Reaction templates
│   ├── warhead_dealed.txt             # Warhead library
│   ├── e3_ligand_dealed.txt           # E3 ligand library
│   └── testset.csv                    # Test set
├── pretrained_models/                 # Pretrained checkpoints
│   └── synprotac_prior.ckpt
├── synprotac/                         # 🔧 Core Python package
│   ├── chem/                          # Chemical utilities (fingerprints, molecules)
│   ├── chemistry/                     # Reaction planning, MCTS, synthesis interface
│   ├── data/                          # Dataset and data modules
│   ├── models/                        # Neural network models
│   │   ├── transformer/               # Graph transformer modules
│   │   ├── scores/                    # Scoring functions (docking, ROCS, etc.)
│   │   ├── encoder.py                 # Graphormer encoder
│   │   ├── decoder.py                 # Transformer decoder
│   │   ├── lightning_module.py        # PyTorch Lightning training module
│   │   ├── rl_module.py               # RL training module
│   │   └── ...
│   ├── retrosynthesis/                # Retrosynthetic analysis
│   └── utils/                         # Helper functions
├── scripts/                           # 📜 Training and evaluation scripts
│   ├── train/                         # Model training
│   ├── sample/                        # Molecule generation
│   ├── eval/                          # Evaluation
│   ├── rl/                            # Reinforcement learning
│   │   ├── 2D_similarity/             # 2D similarity guided RL
│   │   └── Constrained_Docking/       # Docking guided RL
│   ├── rl_without_Crystal_struct/     # 🆕 RL without crystal structure
│   │   ├── CDK9/                      # CDK9 example
│   │   ├── VHL-FKBP5/                 # VHL-FKBP5 example
│   │   ├── VHL-SMARCA2/               # VHL-SMARCA2 example
│   │   └── WDR5-VHL1/                 # WDR5-VHL1 example
│   └── mcts_search/                   # MCTS-based datasets creation
├── bp/                                # Backup / legacy versions
├── environment.yaml                   # Conda environment specification
├── setup.py                           # Package installation
└── README.md                          # This file
```

---

## 📚 Citation

If you use SynPROTAC in your research, please cite:

```bibtex
@article{synprotac,
  title={Design synthesizable PROTACs through synthesis constrained generative model and reinforcement learning},
  author={Xu, Mingyuan and others},
  journal={...},
  year={2024}
}
```

## 📄 License

This project is licensed under the **GPLv3** License. See the `LICENSE` file for details.

---

> **Questions or issues?** Please open an issue on [GitHub](https://github.com/MingyuanXu/SynPROTAC/issues). 


