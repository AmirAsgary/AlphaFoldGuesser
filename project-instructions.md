# Initial Guess (IG) Pipeline Expansion — Project Instructions

## Context & Role

You are an expert in computational structural biology, AlphaFold2 internals, and JAX/Haiku-based ML systems. You are working on expanding a modified AlphaFold2 codebase called **AFfine** (based on `alphafold_finetune` by Phil Bradley, with modifications by Amir Asgary). The codebase lives in the `AFfine/` folder and contains AlphaFold source code plus custom utilities for a method called **Initial Guess (IG)**.

## How Initial Guess Actually Works (Critical — Do Not Get This Wrong)

IG does **NOT** feed coordinates directly to the AlphaFold structure module. The structure module always starts from zero affine (the "black hole") via `generate_new_affine()` in `folding.py`. Here is the actual mechanism:

1. **IG coordinates are placed into `prev_pos`** in `AlphaFold.__call__()` (`modules.py`):
   - `prev_pos = jnp.zeros([N_res, atom_type_num, 3])`
   - `if initial_guess: prev_pos += initial_guess`
   - This `prev_pos` is bundled into the `prev` dict alongside zero `prev_msa_first_row` and zero `prev_pair`.

2. **`prev_pos` enters the RecyclingEmbedder** inside `EmbeddingsAndEvoformer.__call__()` (`modules.py`):
   - `pseudo_beta_fn(aatype, prev_pos)` extracts CB positions (CA for glycine).
   - `dgram_from_positions()` computes pairwise distance bins (histogram).
   - A single `Linear(pair_channel)` projects the dgram.
   - This is **added to `pair_activations`** before the 48 evoformer blocks.
   - **The 3D coordinates are lost here** — only inter-residue distances survive.

3. **The 48 evoformer blocks** process these pair activations (plus template embeddings, MSA, relative position encoding) through triangle attention and multiplication. The output `representations['single']` and `representations['pair']` encode the IG distance information deeply but nonlinearly.

4. **The structure module** (`generate_affines()` in `folding.py`) receives these representations and folds from zero affine. It has never seen the IG coordinates. It relies entirely on the evoformer representations to guide folding via IPA (Invariant Point Attention).

5. **After recycle 0**, the structure module's output replaces `prev_pos`. The original IG coordinates are overwritten. Subsequent recycles use AF's own predictions. IG only directly influences recycle 0.

### IG vs Template Input

Both feed into `pair_activations` before the evoformer, but differ significantly:

- **IG path (RecyclingEmbedder):** coords → CB/CA → pairwise distance bins → one Linear layer → pair_activations. Distances only, no orientation, no sequence info. Influences recycle 0 only. Uses weights trained for "own previous prediction."
- **Template path (SingleTemplateEmbedding):** coords → CB dgram + backbone N,CA,C affine transforms + inter-residue unit vectors + aatype one-hot → full TemplatePairStack (mini-evoformer with triangle attention) → attention-weighted across templates → pair_activations. Provides distances + orientations + sequence identity. Persists across ALL recycles. Also injects torsion angle embeddings into MSA rows.

### Confidence Head Inputs (Critical for Task 4)

- **pLDDT head** reads `representations['structure_module']` — the internal `act` tensor from the final FoldIteration layer. This varies with structure module input.
- **PAE head** reads `representations['pair']` — the evoformer pair output. This does NOT depend on structure module output.
- Both are simple linear networks (LayerNorm → Linear → ReLU → Linear), not expensive to run.

## Current Codebase Architecture (Key Files)

- **`run_prediction.py`**: CLI entry point. Parses args, loads targets TSV, reads `template_pdb_dict` JSON, calls `load_model_runners()`, iterates targets, calls `run_alphafold_prediction()`.
- **`predict_utils.py`**: Contains `load_model_runners()`, `initial_guess_features()`, `run_alphafold_prediction()`, `predict_structure()`, `create_single_template_features()`, `compile_template_features()`. Sets `config.model.embeddings_and_evoformer.initial_guess = True/False`.
- **`af2_util.py`**: Contains `get_atom_positions_initial_guess()`, `get_atom_positions_from_pdb()`, `parse_initial_guess()`, `get_pdb_sequence()`. Currently has **hardcoded pMHC-specific logic** (anchor positions, peptide length, MHC-I intermediate anchors, core region zeroing).
- **`alphafold/model/modules.py`**: `AlphaFold.__call__()` (recycling + IG injection into prev_pos), `AlphaFoldIteration.__call__()` (runs evoformer + all heads), `EmbeddingsAndEvoformer.__call__()` (RecyclingEmbedder, template embedding, evoformer stack), `SingleTemplateEmbedding`, `TemplateEmbedding`, `PredictedLDDTHead`, `PredictedAlignedErrorHead`.
- **`alphafold/model/folding.py`**: `generate_affines()` (creates zero affine, runs FoldIteration loop), `FoldIteration` (IPA + backbone update), `StructureModule` (head wrapper).
- **`alphafold/model/model.py`**: `RunModel` class with `predict()`, `apply_predict` (JIT-compiled). Passes `initial_guess` through to `AlphaFold.__call__()`.
- **`alphafold/model/config.py`**: Contains `initial_guess: False` flag under `embeddings_and_evoformer`.

## Token System (Unified Mask Convention)

A single numpy array `mask_array` of shape `[N_res]` with integer tokens controls behavior across the entire pipeline. This is general-purpose — works for any protein complex, not just pMHC.

| Token | Name | IG Effect | Template Evo Effect | Sampling Effect (Task 4) |
|-------|------|-----------|-------------------|------------------------|
| **0** | Default / Keep | Coords preserved | Template preserved | Can be sampled if within radius of a sampling center |
| **-1** | Mask (always) | Coords → 0 | Template → 0 (optional) | Always zeroed in repr. Excluded from radius neighbor sets |
| **-2** | Sampling Center | Coords preserved (used for radius calc) | Template preserved or masked | Center of sampling sphere. Stochastically zeroed per sample |
| **+1** | Stable (never sample) | Coords preserved | Template preserved | NEVER zeroed even if inside sampling radius. Immutable. |

### Backward Compatibility with pMHC Anchors

The old pMHC behavior maps to: MHC residues → 0 or +1, peptide anchors → 0, non-anchor peptide → -1. A conversion function should translate legacy (anchors, peptide_seq, mhc_len) inputs into the new mask_array format.

---

## Tasks to Implement

### Task 1: Flexible Coordinate Input (PDB or Numpy Arrays, CA-only or CA+CB)

**Goal:** Accept structural input as either a PDB file or raw numpy arrays. Support CA-only or CA+CB mode. Handle multi-chain proteins with chain breaks. Works for any protein complex. Both the IG path and the evoformer template path should use the same unified input parsing.

#### Core Input Parser

**NEW function `parse_structure_input()`** in `af2_util.py`:

This is the central parsing function. It accepts EITHER a PDB file OR numpy arrays, and produces standardized AF2-compatible atom position arrays.

```
Input options (two modes):

  MODE A — PDB file:
    pdb_path: str                          # path to PDB file
    use_only_CA: bool = False              # if True, extract only CA (treat as artificial)
    use_CA_CB: bool = False                # if True, extract CA and CB only
    # If both False, extract all available atoms (full PDB mode)
    # Chain IDs, sequence, residue indices, chain breaks all parsed from PDB automatically

  MODE B — Numpy arrays:
    coords: np.ndarray                     # [N_res, 3] for CA-only, or [N_res, 37, 3] for full atoms
    sequence: str                          # full amino acid sequence (one-letter, all chains concatenated)
    chain_ids: np.ndarray[str]             # [N_res] chain ID per residue, e.g. ['A','A','A','B','B']
    residue_indices: np.ndarray[int]       # [N_res] residue index per residue, e.g. [1,2,3,1,2]
    use_only_CA: bool = False              # if True, input coords is [N_res, 3] (CA only)
    use_CA_CB: bool = False                # if True, input coords is [N_res, 2, 3] (CA+CB)
    # If both False, coords must be [N_res, 37, 3] (full atom)

Output:
    all_positions: np.ndarray [N_res, 37, 3]       # atom coordinates in AF2 atom order
    all_positions_mask: np.ndarray [N_res, 37]      # 1.0 for present atoms, 0.0 for absent
    sequence: str                                    # parsed or pass-through sequence
    chain_ids: np.ndarray[str] [N_res]              # parsed or pass-through chain IDs
    residue_indices: np.ndarray[int] [N_res]        # parsed or pass-through residue indices
```

**Behavior details:**

- **PDB mode:** Parse PDB for ATOM records. Extract chain IDs, residue numbers, atom names, coordinates. Handle altLoc (prefer ' ', 'A', '1'). Handle MSE→MET. Detect chain breaks by: (a) chain ID changes, and (b) CA-CA distance > 4.0Å within same chain. If `use_only_CA=True`: keep only CA positions, zero everything else, set mask accordingly. If `use_CA_CB=True`: keep CA and CB, zero rest. If neither: keep all parsed atoms.

- **Array mode with `use_only_CA=True`:** Input coords is `[N_res, 3]`. Place at `atom_order['CA']`. For non-GLY residues, estimate CB position (~1.5Å from CA along a canonical direction or small random offset). Set mask=1 for CA (and CB if estimated). All other atoms = 0.

- **Array mode with `use_CA_CB=True`:** Input coords is `[N_res, 2, 3]` where `[:,0,:]` = CA, `[:,1,:]` = CB. Place at appropriate atom_order indices. Set masks accordingly.

- **Chain break handling:** Compute `residue_index` array with +200 offsets at chain boundaries (matching existing AFfine convention). Chain breaks detected from `chain_ids` transitions. Return this as part of output or apply downstream.

- **Robustness:** Validate shapes, handle missing CB for GLY gracefully (use CA as pseudo-beta), handle non-standard residues, handle gaps in residue numbering.

#### IG Path Integration

**MODIFY `initial_guess_features()`** in `predict_utils.py`:
- Accept output of `parse_structure_input()` directly.
- If `mask_array` is provided (Task 2), apply `apply_ig_mask()` after parsing.
- Call `parse_initial_guess()` at the end (format conversion to jnp).
- Remove old `template_pdb_path` + `aln` + `anchors` + `peptide_seq` path, or keep as legacy wrapper that internally calls the new path.

#### Evoformer Template Path Integration

**NEW function `create_template_features_from_structure()`** in `predict_utils.py`:

Uses output of `parse_structure_input()` to build AF2-compatible template features.

```
Input:
    all_positions, all_positions_mask, sequence, chain_ids, residue_indices
    (from parse_structure_input)

Output:
    template feature dict:
      template_all_atom_positions: [1, N_res, 37, 3]
      template_all_atom_masks: [1, N_res, 37]
      template_aatype: [1, N_res, 22]  (one-hot via HHBLITS_AA_TO_ID)
      template_pseudo_beta: [1, N_res, 3]
      template_pseudo_beta_mask: [1, N_res]
      template_domain_names: ['none'.encode()]
      template_sequence: [sequence.encode()]
      template_sum_probs: [1, 1] (zeros or ones)
```

- `template_pseudo_beta` computed from CA/CB (use CA for GLY, CB for others; if CB missing, use CA).
- When only CA is available: backbone affine features in `SingleTemplateEmbedding` are automatically masked by existing code (`template_mask = mask[N] * mask[CA] * mask[C]` → zero when N or C missing). The dgram from pseudo_beta still contributes distance information. No changes needed in `modules.py`.
- Should integrate with existing `compile_template_features()` — can be mixed with real PDB-based templates in the stack.

#### CLI Changes

```
--ig_pdb PATH                  # PDB file for IG input
--ig_coords_npy PATH           # numpy array for IG coordinates
--ig_use_only_CA               # flag: extract/use only CA atoms
--ig_use_CA_CB                 # flag: extract/use CA and CB atoms
--ig_sequence STR              # sequence (required for array mode)
--ig_chain_ids_npy PATH        # chain ID array (required for array mode)
--ig_residue_indices_npy PATH  # residue index array (required for array mode)
--template_from_ig              # flag: also use IG structure as evoformer template
```

### Task 2: General IG Masking via Token Array

**Goal:** Replace hardcoded pMHC anchor/peptide zeroing logic with the general token-based masking system. Works for any protein complex. User-friendly with automatic defaults, but fully customizable for advanced use.

#### Automatic Mask Generation

**NEW function `generate_mask_from_structure()`** in `af2_util.py` or `sampling_utils.py`:

For user-friendly operation, automatically generate a mask array from structural information when the user only specifies minimal inputs.

```
Input:
    all_positions: np.ndarray [N_res, 37, 3]     # from parse_structure_input
    chain_ids: np.ndarray[str] [N_res]            # chain IDs
    residue_indices: np.ndarray[int] [N_res]      # residue indices
    mask_residues: Optional[list or np.ndarray]    # residue specs to mask (-1), e.g. ["B:5-9"] or [185,186,187]
    sampling_centers: Optional[list or np.ndarray] # residue specs for sampling centers (-2)
    stable_residues: Optional[list or np.ndarray]  # residue specs for stable tokens (+1)
    auto_sampling_radius: Optional[float]          # if set, auto-assign sampling centers around mask residues

Output:
    mask_array: np.ndarray[int] [N_res]  # tokens: 0, -1, -2, +1
```

**Automatic mode behaviors:**
- If only `mask_residues` is provided: set those to -1, everything else to 0.
- If `mask_residues` + `auto_sampling_radius` provided: set mask residues to -1, auto-detect neighbors within radius and set them to -2 (sampling centers), everything else to 0.
- If `mask_residues` + `auto_sampling_radius` + `stable_residues` provided: additionally set stable positions to +1 (these override any auto-assigned -2 within radius).
- **Residue specification format:** Accept flexible formats: raw index arrays, chain:range strings like `"B:5-9"`, or `"A:*"` for entire chain. Parse via a utility function.

#### Core Masking Function

**NEW function `apply_ig_mask()`** in `af2_util.py`:
- Input: `all_positions [N_res, 37, 3]`, `mask_array [N_res]`
- Output: `masked_positions [N_res, 37, 3]`
- Vectorized: `zero_mask = (mask_array == -1)` → `positions[zero_mask] = 0`. Tokens 0, +1, -2 keep their coords. Only -1 gets zeroed.

#### Codebase Changes

- **MODIFY `get_atom_positions_from_pdb()`** in `af2_util.py`:
  - Remove all pMHC-specific code: `pep_len`, `full_anchors`, `core_region`, `mhc_len` logic, the MHC-I intermediate anchor insertion.
  - Replace with call to `apply_ig_mask()` if `mask_array` is provided.
  - Keep backward compatibility: if `anchors` + `peptide_seq` provided (legacy mode), auto-convert to `mask_array` internally via a `legacy_anchors_to_mask()` conversion function.

- **MODIFY `initial_guess_features()`** in `predict_utils.py`: add `ig_mask: Optional[np.ndarray]` parameter. Pass through. Apply after coordinates are built, before `parse_initial_guess()`.

- **MODIFY `run_prediction.py`**:
  ```
  --ig_mask_npy PATH             # explicit mask array file (advanced users)
  --mask_residues "B:5-9"        # user-friendly: residues to mask
  --sampling_centers "B:3,B:11"  # user-friendly: sampling center residues
  --stable_residues "A:*"        # user-friendly: stable residues
  --auto_sampling_radius 8.0     # auto-generate sampling centers around mask residues
  ```

### Task 3: Evoformer Template Masking

**Goal:** Optionally mask specific residue regions in the template features fed to the evoformer. Independent from IG masking. Studies show this helps predict point mutations. Should support both automatic and manual mask specification.

#### Template Masking Function

**NEW function `apply_template_mask()`** in `predict_utils.py` or `af2_util.py`:
- Input: `template_features: dict`, `evo_mask: np.ndarray [N_res]` (uses same token convention, -1 = mask)
- Output: modified `template_features` dict
- For masked residues (-1): zero out `template_all_atom_positions[:, masked, :, :]`, `template_all_atom_masks[:, masked, :]`, `template_pseudo_beta[:, masked, :]`, `template_pseudo_beta_mask[:, masked]`.
- Existing `SingleTemplateEmbedding` handles this automatically: `template_mask_2d` becomes zero for masked rows/columns, and `act *= template_mask_2d[..., None]` zeros the entire masked region before `TemplatePairStack`.

#### Automatic vs Manual Control

**Automatic mode:** If `--evo_template_mask auto` is set, reuse the same mask_array from Task 2. Residues that are -1 (mask) in IG are also masked in the template. This is the simplest user experience — one mask controls everything.

**Independent mode:** If `--evo_template_mask_npy PATH` is provided, use a separate mask array for template masking. This allows combinations like: mask position 150 in IG but keep it in the template (AF sees template distance info but must predict from zero IG coords), or vice versa.

**4 combinations per residue region:**
- IG kept + template kept → AF has both IG distance hint and template structural info
- IG masked + template kept → AF has no recycling distance hint, but template provides structural context
- IG kept + template masked → AF has recycling distance hint, but no template signal
- Both masked → fully free prediction from sequence/MSA only

#### Codebase Changes

- **MODIFY `predict_structure()`** in `predict_utils.py`: apply `apply_template_mask()` after `compile_template_features()` and before `model_runner.process_features()`.

- **MODIFY `run_prediction.py`**:
  ```
  --evo_template_mask auto        # reuse IG mask for templates
  --evo_template_mask_npy PATH    # separate template mask (advanced)
  ```

### Task 4: Sampling Pipeline via Evoformer Representation Masking

**Goal:** Run evoformer ONCE with IG, then run structure module N times by stochastically masking cached evoformer representations. Get distributions over structure, pLDDT, and PAE. General-purpose for any protein complex.

**Core insight:** Since the structure module always starts from zero affine and takes evoformer representations as input, we can cache the evoformer output and repeatedly run the structure module with different masked versions of those representations. No evoformer re-run. Fully within AF's trained behavior.

#### Phase A: One Full AF Run with IG (runs once)

Run the normal IG pipeline (all recycles). IG shapes the representations through the full recycling process. After the final evoformer pass, cache:

- `cached_single`: `representations['single']` — shape `[N_res, c_s]`
- `cached_pair`: `representations['pair']` — shape `[N_res, N_res, c_z]`

**Implementation:** In `AlphaFoldIteration.__call__()` (`modules.py`), the `representations` dict already contains `single` and `pair` after the evoformer stack. The existing `--return_all_outputs` flag already returns representations. Extend this to support caching for sampling mode.

#### Phase B: Compute Residue Sets (once)

From `mask_array` and IG coordinates. This should be fully automatic from the mask array produced in Task 2.

```
MASK_SET     = {i : mask[i] == -1}   → always zeroed in repr
CENTERS      = {i : mask[i] == -2}   → sampling centers
STABLE_SET   = {i : mask[i] == +1}   → never touched
DEFAULT_SET  = {i : mask[i] == 0}    → eligible if in radius

# Radius computation (vectorized):
all_CA = ig_coords[:, CA_idx, :]                          # [N_res, 3]
center_CA = all_CA[CENTERS]                                # [|centers|, 3]
dists = jnp.linalg.norm(all_CA[None,:] - center_CA[:,None], axis=-1)  # [|centers|, N_res]
in_any_radius = jnp.any(dists <= radius, axis=0)          # [N_res]

# Eligible = in radius AND default token (NOT stable, NOT mask)
ELIGIBLE = in_any_radius & (mask == 0)
SAMPLEABLE = CENTERS ∪ ELIGIBLE
```

**Stable tokens (+1) are excluded from SAMPLEABLE** even if geometrically inside a sampling center's radius. **Mask tokens (-1) are excluded from radius neighbor sets** — they are always zeroed independently.

**Automatic mode:** If user only provided `--mask_residues` and `--auto_sampling_radius` in Task 2, the mask_array and all residue sets are already computed. No additional user input needed for Task 4 beyond `--sampling_mode` and `--n_times_sampling`.

#### Phase C: Per-Sample Stochastic Masking (N times)

For each sample `i` in `1..n_times_sampling`, with independent RNG seeds:

**Single representation masking** (seed_ig_i):
- `masked_single = cached_single.copy()`
- `masked_single[MASK_SET] = 0` (always)
- `ig_sample = random_choice(SAMPLEABLE, fraction=sampling_fraction_IG, seed=seed_ig_i)`
- `masked_single[ig_sample] = 0`

**Pair representation masking** (seed_evo_i ≠ seed_ig_i):
- `masked_pair = cached_pair.copy()`
- `masked_pair[MASK_SET, :] = 0; masked_pair[:, MASK_SET] = 0` (always)
- `evo_sample = random_choice(SAMPLEABLE, fraction=sampling_fraction_evo, seed=seed_evo_i)`
- `masked_pair[evo_sample, :] = 0; masked_pair[:, evo_sample] = 0`

The same SAMPLEABLE residue set is used for both, but different fractions and different random seeds produce different subsets being zeroed.

#### Phase D: Structure Module + Heads Only (per sample)

For each sample, run:
1. `StructureModule(masked_single_i, masked_pair_i, batch)` → `coords_i`, `act_i`
   - `generate_new_affine()` → zeros (trained behavior, no changes to `folding.py`)
   - Structure module sees different masked representations → folds differently for masked/sampled regions
   - Stable regions always get consistent signal → fold consistently
2. `PredictedLDDTHead(act_i)` → `plddt_i [N_res]` — varies per sample (depends on SM activations)
3. `PredictedAlignedErrorHead(masked_pair_i)` → `pae_i [N_res, N_res]` — varies per sample (depends on masked pair)

**Key implementation change in `modules.py`:** Need to expose a method on `AlphaFoldIteration` (or create a new function) that runs only the structure module and confidence heads with externally provided representations, bypassing the evoformer. This is the main architectural change for Task 4.

**Key implementation change in `model.py`:** Add `predict_sampling()` method on `RunModel` that runs the full pipeline once, caches representations, then loops the structure module N times. Consider separate JIT compilation for the structure-only path.

**Parallelization:** Use `jax.vmap` over samples for GPU parallelism where memory allows.

#### Phase E: Distribution Collection

Stack all N samples. Compute:
- Per-residue pLDDT mean, std, percentiles
- Per-pair PAE distributions
- Coordinate RMSF (per-residue flexibility / dynamics)
- Multi-model PDB ensemble (MODEL/ENDMDL records) or npz archive
- Per-residue sampling frequency map (which residues were masked how often)

**Computational cost:** Structure module + heads ≈ 5-10% of full forward pass. For 100 samples: ~6-11× cost of single prediction (vs 100× if re-running full pipeline).

#### CLI Flags for Task 4

```
--sampling_mode                    # enable sampling pipeline
--n_times_sampling 100             # number of structure module samples
--radius 8.0                       # angstrom radius around sampling centers
--sampling_fraction_IG 0.5         # fraction of SAMPLEABLE residues zeroed in single repr
--sampling_fraction_evo 0.3        # fraction zeroed in pair repr (different seed)
```

All masking inputs come from Task 2's CLI (`--ig_mask_npy`, `--mask_residues`, `--sampling_centers`, `--stable_residues`, `--auto_sampling_radius`). Task 4 consumes the mask_array automatically.

#### User-Friendly Usage Examples

**Simplest case — mutation study:**
```bash
python run_prediction.py \
  --targets targets.tsv \
  --ig_pdb complex.pdb \
  --mask_residues "B:150" \
  --auto_sampling_radius 10.0 \
  --stable_residues "A:*" \
  --sampling_mode \
  --n_times_sampling 100
```
This automatically: parses PDB → builds IG coords → sets residue 150 of chain B as mask (-1) → finds all chain B neighbors within 10Å as sampling centers (-2) → sets all chain A as stable (+1) → runs evoformer once → samples structure module 100 times.

**Advanced case — full manual control:**
```bash
python run_prediction.py \
  --targets targets.tsv \
  --ig_coords_npy coords.npy \
  --ig_sequence "MKTLLILAVL..." \
  --ig_chain_ids_npy chains.npy \
  --ig_residue_indices_npy residx.npy \
  --ig_use_only_CA \
  --ig_mask_npy mask.npy \
  --evo_template_mask_npy tmpl_mask.npy \
  --sampling_mode \
  --n_times_sampling 500 \
  --radius 12.0 \
  --sampling_fraction_IG 0.6 \
  --sampling_fraction_evo 0.4
```

---

## Implementation Order

**Task 1 → Task 2 → Task 3 → Task 4**

- Task 1 is independent and provides the unified input parsing foundation.
- Task 2 depends on Task 1's parsing output and introduces the token system required by Task 4.
- Task 3 is independent of Tasks 1-2 but provides template masking used in Task 4.
- Task 4 depends on Task 2 (token system) and is the most complex.

## Files That Change

| File | Change Type | Tasks |
|------|-------------|-------|
| `run_prediction.py` | MODIFY (CLI args, sampling orchestration) | 1, 2, 3, 4 |
| `predict_utils.py` | MODIFY + NEW functions | 1, 2, 3, 4 |
| `af2_util.py` | MODIFY + NEW functions | 1, 2 |
| `alphafold/model/modules.py` | MODIFY (evoformer caching, structure-only execution) | 4 |
| `alphafold/model/model.py` | MODIFY (predict_sampling method) | 4 |
| `alphafold/model/config.py` | MODIFY (sampling config) | 4 |
| `alphafold/model/folding.py` | **NO CHANGES** | — |
| NEW: `sampling_utils.py` | NEW (token parsing, radius, stochastic masking, residue spec parsing, stats) | 2, 4 |

## Code Quality Requirements

- Always comment your code.
- Do not put gaps between code lines.
- Write function descriptions (docstrings with Args/Returns).
- Make code functional and tested.
- Vectorize all operations — use jax/numpy broadcasting, avoid Python loops.
- Memory efficient — stream data where possible.
- In JAX/TF operations, minimize Python-level loops and function call overhead.
- Use `jax.vmap` for parallelism over samples in Task 4.
- All mask operations must be vectorized (no per-residue Python loops).
- Parsing functions must be robust to edge cases: missing atoms, non-standard residues, chain breaks, single-chain proteins, multi-chain complexes, empty chains.
