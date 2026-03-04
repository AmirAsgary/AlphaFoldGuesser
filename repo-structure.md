# AFfine-IG — Repository Structure & README

---

## Folder & File Layout

```
AFfine/
│
├── README.md                          # ← this file
├── requirements.txt                   # Python dependencies
├── setup.py                           # (optional) package install
│
│── ── ── ── ── ENTRY POINTS ── ── ── ── ──
├── run_prediction.py                  # CLI entry (MODIFIED — new IG args)
├── run_finetuning.py                  # Training entry (unchanged)
│
│── ── ── ── ── IG PIPELINE (NEW) ── ── ── ── ──
├── ig_pipeline.py                     # Tasks 1-3: input parsing, masking,
│                                      #   template masking, feature builders
├── sampling_utils.py                  # Task 4: stochastic sampling pipeline
├── predict_utils_ig.py                # Updated prediction functions (v2)
├── run_prediction_ig_patch.py         # CLI arg definitions + integration glue
│
│── ── ── ── ── EXISTING UTILITIES ── ── ── ── ──
├── predict_utils.py                   # Original prediction utilities (kept)
├── af2_util.py                        # Original AF2 helpers (kept for legacy)
├── train_utils.py                     # Training helpers (unchanged)
│
│── ── ── ── ── ALPHAFOLD CORE ── ── ── ── ──
├── alphafold/
│   ├── __init__.py
│   ├── common/
│   │   ├── protein.py                 # Protein dataclass, PDB I/O
│   │   ├── residue_constants.py       # Atom orders, amino acid mappings
│   │   └── confidence.py              # Confidence metric utilities
│   ├── data/
│   │   ├── pipeline.py                # Feature construction (MSA, seq)
│   │   ├── templates.py               # Template feature specs
│   │   ├── mmcif_parsing.py           # mmCIF parser
│   │   └── tf/
│   │       └── proteins_dataset.py    # TF dataset utilities
│   ├── model/
│   │   ├── model.py                   # RunModel class (MODIFIED — IG pass-through)
│   │   ├── modules.py                 # AlphaFold, Evoformer, heads (MODIFIED — IG injection)
│   │   ├── folding.py                 # Structure module (NO CHANGES)
│   │   ├── config.py                  # Model config (MODIFIED — initial_guess flag)
│   │   ├── all_atom.py                # Atom-level ops
│   │   ├── common_modules.py          # Linear, LayerNorm, etc.
│   │   ├── layer_stack.py             # Evoformer layer stacking
│   │   ├── prng.py                    # PRNG helpers
│   │   ├── utils.py                   # Misc JAX utilities
│   │   ├── features.py                # Feature processing
│   │   ├── data.py                    # Param loading
│   │   └── tf/
│   │       └── ...
│   └── relax/
│       ├── amber_minimize.py          # OpenMM relaxation
│       └── ...
│
│── ── ── ── ── DATA & EXAMPLES ── ── ── ── ──
├── examples/
│   ├── mutation_study/                # Example: single-mutation prediction
│   │   ├── complex.pdb
│   │   ├── targets.tsv
│   │   └── run.sh
│   ├── sampling_demo/                 # Example: sampling pipeline
│   │   ├── complex.pdb
│   │   ├── mask.npy
│   │   ├── targets.tsv
│   │   └── run.sh
│   └── pmhc_legacy/                   # Legacy pMHC examples (backward compat)
│       └── ...
│
│── ── ── ── ── TESTS ── ── ── ── ──
├── tests/
│   ├── test_ig_pipeline.py            # Tests for parse_structure_input, masking
│   ├── test_sampling_utils.py         # Tests for residue sets, mask generation
│   └── test_integration.py            # End-to-end prediction tests
│
│── ── ── ── ── OUTPUTS (generated) ── ── ── ── ──
└── outputs/                           # Created at runtime
    ├── *_model_1_*.pdb                # Ranked structure predictions
    ├── *_plddt.npy                    # Per-residue pLDDT
    ├── *_sampling_results.npz         # Sampling distributions (Task 4)
    ├── *_ensemble.pdb                 # Multi-model ensemble (Task 4)
    ├── *_plddt_mean.npy               # Mean pLDDT across samples
    ├── *_coord_rmsf.npy               # Per-residue flexibility
    └── ...
```

---

## Which file does what

| File | Role | Tasks |
|---|---|---|
| `ig_pipeline.py` | Core library — parsing, masking, template features | 1, 2, 3 |
| `sampling_utils.py` | Stochastic sampling loop + output writers | 4 |
| `predict_utils_ig.py` | Updated `predict_structure_v2`, `run_alphafold_prediction_v2` | 1–4 |
| `run_prediction_ig_patch.py` | CLI argument definitions + per-target orchestration | 1–4 |
| `run_prediction.py` | Entry point (patched to import from `*_ig_patch`) | — |
| `alphafold/model/modules.py` | IG injection into `prev_pos`, representation caching | 4 |
| `alphafold/model/model.py` | `RunModel.predict()` passes IG through | — |
| `alphafold/model/folding.py` | Structure module — **never modified** | — |

---

# README

## AFfine-IG: General-Purpose Initial Guess Pipeline for AlphaFold2

An expansion of [AFfine](https://github.com/phbradley/alphafold_finetune) that
replaces hardcoded pMHC logic with a general-purpose Initial Guess (IG) system.
Works with **any protein complex** — not just pMHC.

### What it does

1. **Flexible structure input** — feed IG coordinates via PDB or numpy arrays,
   with CA-only, CA+CB, or full-atom modes.
2. **Token-based masking** — a single `mask_array` controls which residues are
   masked, sampled, or held stable across the entire pipeline.
3. **Template masking** — optionally mask regions in the evoformer template path
   independently from the IG path.
4. **Stochastic sampling** — run the evoformer once, then sample the structure
   module N times with randomly masked representations to get distributions
   over structure, pLDDT, and PAE.

### Installation

```bash
# Clone the repo
git clone https://github.com/youruser/AFfine.git
cd AFfine

# Install dependencies (JAX with GPU, TensorFlow, BioPython, etc.)
pip install -r requirements.txt

# Download AlphaFold parameters into a data/ directory
# data/params/ should contain model_*.npz files
```

### Quick Start

**Simplest case — mutation study with auto-masking:**

```bash
python run_prediction.py \
  --targets targets.tsv \
  --data_dir ./data \
  --model_names model_2_ptm \
  --ig_pdb complex.pdb \
  --mask_residues "B:150" \
  --auto_sampling_radius 10.0 \
  --stable_residues "A:*"
```

This parses the PDB, masks residue 150 on chain B, auto-detects neighbors
within 10 Å as sampling centers, and locks chain A as stable.

**With stochastic sampling (100 structure module runs):**

```bash
python run_prediction.py \
  --targets targets.tsv \
  --data_dir ./data \
  --model_names model_2_ptm \
  --ig_pdb complex.pdb \
  --mask_residues "B:150" \
  --auto_sampling_radius 10.0 \
  --stable_residues "A:*" \
  --sampling_mode \
  --n_times_sampling 100
```

Outputs: `*_ensemble.pdb` (100-model PDB), `*_plddt_mean.npy`,
`*_coord_rmsf.npy`, `*_sampling_results.npz`.

**Full manual control with numpy arrays:**

```bash
python run_prediction.py \
  --targets targets.tsv \
  --data_dir ./data \
  --model_names model_2_ptm \
  --ig_coords_npy coords.npy \
  --ig_sequence "MKTLLILAVL..." \
  --ig_chain_ids_npy chains.npy \
  --ig_residue_indices_npy residx.npy \
  --ig_use_only_CA \
  --ig_mask_npy mask.npy \
  --evo_template_mask auto \
  --sampling_mode \
  --n_times_sampling 500 \
  --radius 12.0 \
  --sampling_fraction_IG 0.6 \
  --sampling_fraction_evo 0.4
```

**Legacy pMHC mode (backward compatible — no new flags needed):**

```bash
python run_prediction.py \
  --targets targets.tsv \
  --data_dir ./data \
  --model_names model_2_ptm
```

Works exactly as before when no `--ig_*` flags are provided.

### Token System

The mask array `[N_res]` uses four integer tokens:

| Token | Meaning | IG coords | Template | Sampling |
|-------|---------|-----------|----------|----------|
| `0`   | Default | kept      | kept     | eligible if near a center |
| `-1`  | Mask    | zeroed    | zeroed (optional) | always zeroed in repr |
| `-2`  | Center  | kept      | kept     | stochastically zeroed |
| `+1`  | Stable  | kept      | kept     | never touched |

### Residue Specification Format

All `--mask_residues`, `--sampling_centers`, `--stable_residues` flags accept:

- `"B:150"` — chain B, residue 150
- `"B:5-9"` — chain B, residues 5 through 9
- `"A:*"` — entire chain A
- `"B:5,B:7,B:10"` — multiple specific residues
- `"42,43,44"` — raw 0-indexed positions (no chain prefix)

### CLI Reference

| Flag | Task | Description |
|------|------|-------------|
| `--ig_pdb` | 1 | PDB file for IG coordinates |
| `--ig_coords_npy` | 1 | Numpy array for IG coordinates |
| `--ig_use_only_CA` | 1 | CA-only mode |
| `--ig_use_CA_CB` | 1 | CA+CB mode |
| `--ig_sequence` | 1 | Sequence (required for array mode) |
| `--ig_chain_ids_npy` | 1 | Chain IDs array (required for array mode) |
| `--ig_residue_indices_npy` | 1 | Residue indices array (required for array mode) |
| `--template_from_ig` | 1 | Also use IG structure as evoformer template |
| `--ig_mask_npy` | 2 | Pre-computed mask array file |
| `--mask_residues` | 2 | Residue spec → token -1 |
| `--sampling_centers` | 2 | Residue spec → token -2 |
| `--stable_residues` | 2 | Residue spec → token +1 |
| `--auto_sampling_radius` | 2 | Auto-detect centers near mask (Å) |
| `--evo_template_mask` | 3 | `"auto"` or path to mask .npy |
| `--sampling_mode` | 4 | Enable sampling pipeline |
| `--n_times_sampling` | 4 | Number of SM samples (default: 100) |
| `--radius` | 4 | Sampling sphere radius in Å (default: 8.0) |
| `--sampling_fraction_IG` | 4 | Fraction zeroed in single repr (default: 0.5) |
| `--sampling_fraction_evo` | 4 | Fraction zeroed in pair repr (default: 0.3) |

### Output Files

| File | Description |
|------|-------------|
| `*_model_1_*.pdb` | Top-ranked predicted structure |
| `*_plddt.npy` | Per-residue pLDDT confidence |
| `*_predicted_aligned_error.npy` | Predicted aligned error matrix |
| `*_ensemble.pdb` | Multi-model PDB from sampling |
| `*_sampling_results.npz` | Full sampling archive (coords, pLDDT, PAE) |
| `*_plddt_mean.npy` | Mean pLDDT across samples |
| `*_plddt_std.npy` | pLDDT standard deviation |
| `*_coord_rmsf.npy` | Per-residue CA RMSF (flexibility) |
| `*_pae_mean.npy` | Mean PAE across samples |
| `*_sampling_freq.npy` | Per-residue masking frequency |

### How to integrate with existing run_prediction.py

Add three lines near the top of `run_prediction.py`:

```python
from run_prediction_ig_patch import (
    add_ig_pipeline_args,
    setup_ig_pipeline,
    process_target_with_ig_pipeline,
)

# After creating the parser, before parse_args():
parser = add_ig_pipeline_args(parser)
args = parser.parse_args()

# After loading model_runners:
ig_config = setup_ig_pipeline(args)

# Inside the per-target loop, replace the prediction call:
all_metrics = process_target_with_ig_pipeline(
    args, ig_config, targetl, query_sequence, query_chainseq,
    all_template_features, model_runners, outfile_prefix,
    crop_size, msa, deletion_matrix,
)
```

### Citation

Based on [alphafold_finetune](https://github.com/phbradley/alphafold_finetune)
by Phil Bradley, with Initial Guess modifications by Amir Asgary.
