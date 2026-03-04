"""
tests/test_sampling_utils.py — Unit tests for sampling_utils.py (Task 4)
=========================================================================
Covers: compute_residue_sets, generate_masked_representations_single_sample,
        generate_all_masked_representations, write_ensemble_pdb.
Run:  pytest tests/test_sampling_utils.py -v
Note: Tests that require JAX are marked and will be skipped if JAX is
      not available (e.g. in a CPU-only CI environment without GPU).
"""
import os
import sys
import tempfile
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ig_pipeline import (
    TOKEN_DEFAULT, TOKEN_MASK, TOKEN_CENTER, TOKEN_STABLE,
    CA_IDX, ATOM_TYPE_NUM,
)
from sampling_utils import (
    compute_residue_sets,
    write_ensemble_pdb,
)
# Conditional JAX import — skip JAX tests if not available
try:
    import jax
    import jax.numpy as jnp
    from sampling_utils import (
        generate_masked_representations_single_sample,
        generate_all_masked_representations,
        _random_subset,
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _make_linear_positions(n_res, spacing=3.8):
    """Build [N_res, 37, 3] with CA atoms along x-axis."""
    pos = np.zeros((n_res, ATOM_TYPE_NUM, 3), dtype=np.float32)
    pos[:, CA_IDX, 0] = np.arange(n_res) * spacing
    return pos
# ─────────────────────────────────────────────────────────────────────
# Phase B: compute_residue_sets
# ─────────────────────────────────────────────────────────────────────
class TestComputeResidueSets:
    """Tests for the vectorized residue set computation."""
    def test_basic_sets(self):
        """Tokens map to correct boolean sets."""
        mask = np.array([0, -1, -2, 1, 0], dtype=np.int32)
        pos = _make_linear_positions(5)
        sets = compute_residue_sets(mask, pos, radius=100.0)
        assert list(sets['mask_set']) == [False, True, False, False, False]
        assert list(sets['centers']) == [False, False, True, False, False]
        assert list(sets['stable_set']) == [False, False, False, True, False]
    def test_sampleable_is_centers_plus_eligible(self):
        """SAMPLEABLE = centers ∪ (in-radius ∩ default)."""
        # 10 residues, 3.8 Å apart. Center at position 5, radius 5.0 Å.
        n_res = 10
        mask = np.zeros(n_res, dtype=np.int32)
        mask[5] = TOKEN_CENTER
        pos = _make_linear_positions(n_res)
        sets = compute_residue_sets(mask, pos, radius=5.0)
        # Center itself is sampleable
        assert sets['sampleable'][5] == True
        # Neighbors at ±3.8 Å (positions 4 and 6) are within radius → eligible
        assert sets['eligible'][4] == True
        assert sets['eligible'][6] == True
        assert sets['sampleable'][4] == True
        assert sets['sampleable'][6] == True
        # Position 3 is 2*3.8=7.6 Å away → outside radius
        assert sets['eligible'][3] == False
        assert sets['sampleable'][3] == False
    def test_stable_excluded_from_sampleable(self):
        """Stable tokens inside radius are NOT sampleable."""
        n_res = 5
        mask = np.array([0, 1, -2, 0, 0], dtype=np.int32)
        # Position 1 is stable, position 2 is center
        pos = _make_linear_positions(n_res, spacing=3.0)  # all within 12 Å
        sets = compute_residue_sets(mask, pos, radius=20.0)
        # Position 1 is stable → not sampleable even though within radius
        assert sets['sampleable'][1] == False
        assert sets['stable_set'][1] == True
    def test_mask_excluded_from_sampleable(self):
        """Mask tokens are not sampleable (they're handled separately)."""
        mask = np.array([-1, -2, 0], dtype=np.int32)
        pos = _make_linear_positions(3, spacing=2.0)
        sets = compute_residue_sets(mask, pos, radius=100.0)
        assert sets['sampleable'][0] == False  # mask token
        assert sets['mask_set'][0] == True
    def test_no_centers_empty_sampleable(self):
        """No center tokens → sampleable is empty."""
        mask = np.array([0, 0, -1, 0, 1], dtype=np.int32)
        pos = _make_linear_positions(5)
        sets = compute_residue_sets(mask, pos, radius=10.0)
        assert sets['sampleable'].sum() == 0
    def test_large_radius_catches_all_defaults(self):
        """Very large radius makes all default-token residues eligible."""
        n_res = 20
        mask = np.zeros(n_res, dtype=np.int32)
        mask[10] = TOKEN_CENTER
        pos = _make_linear_positions(n_res)
        sets = compute_residue_sets(mask, pos, radius=1e6)
        # All default residues should be eligible
        n_default = np.sum(mask == TOKEN_DEFAULT)
        assert sets['eligible'].sum() == n_default
    def test_multiple_centers_union_radius(self):
        """Multiple centers: residue in range of ANY center is eligible."""
        n_res = 10
        mask = np.zeros(n_res, dtype=np.int32)
        mask[2] = TOKEN_CENTER
        mask[7] = TOKEN_CENTER
        pos = _make_linear_positions(n_res, spacing=3.8)
        # radius=4.0 catches ±1 neighbor from each center
        sets = compute_residue_sets(mask, pos, radius=4.0)
        # Neighbors of center at 2: positions 1, 3
        assert sets['eligible'][1] == True
        assert sets['eligible'][3] == True
        # Neighbors of center at 7: positions 6, 8
        assert sets['eligible'][6] == True
        assert sets['eligible'][8] == True
        # Position 5 is 3*3.8=11.4 Å from center 2 and 2*3.8=7.6 from center 7 → outside
        assert sets['eligible'][5] == False
# ─────────────────────────────────────────────────────────────────────
# Phase C: Stochastic masking (JAX)
# ─────────────────────────────────────────────────────────────────────
@requires_jax
class TestRandomSubset:
    """Tests for the JAX random subset selection."""
    def test_respects_sampleable_mask(self):
        """Only sampleable residues can be selected."""
        key = jax.random.PRNGKey(0)
        sampleable = jnp.array([True, False, True, False, True])
        selected = _random_subset(key, sampleable, fraction=1.0)
        # With fraction=1.0, all sampleable should be selected
        assert selected[0] == True
        assert selected[1] == False  # not sampleable
        assert selected[2] == True
        assert selected[3] == False  # not sampleable
        assert selected[4] == True
    def test_fraction_zero_selects_none(self):
        """Fraction 0.0 selects no residues."""
        key = jax.random.PRNGKey(42)
        sampleable = jnp.ones(100, dtype=bool)
        selected = _random_subset(key, sampleable, fraction=0.0)
        assert selected.sum() == 0
    def test_fraction_one_selects_all_sampleable(self):
        """Fraction 1.0 selects all sampleable residues."""
        key = jax.random.PRNGKey(42)
        sampleable = jnp.array([True]*50 + [False]*50)
        selected = _random_subset(key, sampleable, fraction=1.0)
        assert int(selected.sum()) == 50
    def test_approximate_fraction(self):
        """Selected count is approximately fraction * n_sampleable."""
        key = jax.random.PRNGKey(123)
        n = 1000
        sampleable = jnp.ones(n, dtype=bool)
        selected = _random_subset(key, sampleable, fraction=0.3)
        count = int(selected.sum())
        # Should be roughly 300 ± 50 (statistical tolerance)
        assert 200 < count < 400, f"Expected ~300 selected, got {count}"
@requires_jax
class TestMaskedRepresentations:
    """Tests for single-sample masked representation generation."""
    def test_mask_set_always_zeroed(self):
        """Residues in mask_set are always zeroed regardless of sampling."""
        n_res, c_s, c_z = 5, 8, 4
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.array([False, True, False, False, False])  # residue 1 masked
        sampleable = jnp.array([False, False, True, True, False])
        key_ig = jax.random.PRNGKey(0)
        key_evo = jax.random.PRNGKey(1)
        ms, mp = generate_masked_representations_single_sample(
            cached_single, cached_pair,
            mask_set, sampleable,
            sampling_fraction_ig=0.0,  # zero fraction → no stochastic masking
            sampling_fraction_evo=0.0,
            key_ig=key_ig, key_evo=key_evo)
        # Residue 1 (mask_set) should be zeroed in single
        np.testing.assert_array_equal(np.array(ms[1]), 0.0)
        # Residue 1 row and column should be zeroed in pair
        np.testing.assert_array_equal(np.array(mp[1, :, :]), 0.0)
        np.testing.assert_array_equal(np.array(mp[:, 1, :]), 0.0)
        # Non-masked, non-sampled residues should be unchanged
        np.testing.assert_array_equal(np.array(ms[0]), 1.0)
    def test_pair_masking_is_symmetric(self):
        """If a residue is zeroed, both its row AND column are zeroed in pair."""
        n_res, c_s, c_z = 4, 4, 2
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.array([False, True, False, False])
        sampleable = jnp.zeros(n_res, dtype=bool)
        key = jax.random.PRNGKey(0)
        _, mp = generate_masked_representations_single_sample(
            cached_single, cached_pair,
            mask_set, sampleable, 0.0, 0.0, key, key)
        # Row 1 all zero
        assert np.array(mp[1]).sum() == 0.0
        # Column 1 all zero
        assert np.array(mp[:, 1]).sum() == 0.0
        # Entry [0,2] should be nonzero (both row 0 and col 2 are unmasked)
        assert np.array(mp[0, 2]).sum() > 0.0
    def test_different_seeds_different_masks(self):
        """Different PRNG seeds produce different stochastic masks."""
        n_res, c_s, c_z = 50, 8, 4
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.zeros(n_res, dtype=bool)
        sampleable = jnp.ones(n_res, dtype=bool)
        ms1, _ = generate_masked_representations_single_sample(
            cached_single, cached_pair,
            mask_set, sampleable, 0.5, 0.5,
            jax.random.PRNGKey(0), jax.random.PRNGKey(1))
        ms2, _ = generate_masked_representations_single_sample(
            cached_single, cached_pair,
            mask_set, sampleable, 0.5, 0.5,
            jax.random.PRNGKey(99), jax.random.PRNGKey(100))
        # The two samples should differ (statistically certain with 50 residues)
        assert not np.allclose(np.array(ms1), np.array(ms2))
@requires_jax
class TestBatchedMaskedRepresentations:
    """Tests for vmap-parallel batch generation."""
    def test_output_shapes(self):
        """Batched output has correct [n_samples, N_res, ...] shapes."""
        n_res, c_s, c_z = 8, 4, 2
        n_samples = 5
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.zeros(n_res, dtype=bool)
        sampleable = jnp.ones(n_res, dtype=bool)
        all_s, all_p = generate_all_masked_representations(
            cached_single, cached_pair,
            mask_set, sampleable,
            0.5, 0.3, n_samples, base_seed=0)
        assert all_s.shape == (n_samples, n_res, c_s)
        assert all_p.shape == (n_samples, n_res, n_res, c_z)
    def test_samples_differ(self):
        """Each sample in the batch has a different mask pattern."""
        n_res, c_s, c_z = 20, 4, 2
        n_samples = 10
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.zeros(n_res, dtype=bool)
        sampleable = jnp.ones(n_res, dtype=bool)
        all_s, _ = generate_all_masked_representations(
            cached_single, cached_pair,
            mask_set, sampleable,
            0.5, 0.3, n_samples, base_seed=42)
        # Check that not all samples are identical
        all_same = all(np.allclose(np.array(all_s[0]), np.array(all_s[i]))
                       for i in range(1, n_samples))
        assert not all_same, "All samples are identical — RNG not working"
    def test_mask_set_zeroed_in_all_samples(self):
        """mask_set residues are zeroed across every sample."""
        n_res, c_s, c_z = 6, 4, 2
        n_samples = 8
        cached_single = jnp.ones((n_res, c_s))
        cached_pair = jnp.ones((n_res, n_res, c_z))
        mask_set = jnp.array([False, True, False, False, True, False])
        sampleable = jnp.array([True, False, True, True, False, True])
        all_s, all_p = generate_all_masked_representations(
            cached_single, cached_pair,
            mask_set, sampleable,
            0.5, 0.3, n_samples, base_seed=7)
        # Residues 1 and 4 should be zero in ALL samples
        for i in range(n_samples):
            np.testing.assert_array_equal(np.array(all_s[i, 1, :]), 0.0)
            np.testing.assert_array_equal(np.array(all_s[i, 4, :]), 0.0)
            np.testing.assert_array_equal(np.array(all_p[i, 1, :, :]), 0.0)
            np.testing.assert_array_equal(np.array(all_p[i, :, 4, :]), 0.0)
# ─────────────────────────────────────────────────────────────────────
# Phase E: Output writers
# ─────────────────────────────────────────────────────────────────────
class TestWriteEnsemblePdb:
    """Tests for multi-model PDB writer."""
    def test_writes_valid_pdb(self, tmp_path):
        """Output PDB has correct MODEL/ENDMDL/END records."""
        n_samples, n_res = 3, 4
        coords = np.random.randn(n_samples, n_res, ATOM_TYPE_NUM, 3).astype(np.float32)
        # Set most atoms to zero (only CA nonzero) so the PDB isn't huge
        coords[:, :, :, :] = 0.0
        coords[:, :, CA_IDX, :] = np.random.randn(n_samples, n_res, 3)
        aatype = np.zeros(n_res, dtype=np.int32)  # all ALA
        res_idx = np.arange(1, n_res + 1, dtype=np.int32)
        out_path = str(tmp_path / "ensemble.pdb")
        write_ensemble_pdb(coords, aatype, res_idx, out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        # Should have 3 MODEL records
        assert content.count('MODEL') == 3
        assert content.count('ENDMDL') == 3
        assert content.strip().endswith('END')
    def test_plddt_in_bfactor(self, tmp_path):
        """pLDDT scores appear in the B-factor column."""
        n_samples, n_res = 1, 2
        coords = np.zeros((n_samples, n_res, ATOM_TYPE_NUM, 3), dtype=np.float32)
        coords[0, 0, CA_IDX, :] = [1, 2, 3]
        coords[0, 1, CA_IDX, :] = [4, 5, 6]
        aatype = np.zeros(n_res, dtype=np.int32)
        res_idx = np.array([1, 2], dtype=np.int32)
        plddt = np.array([[85.5, 42.3]])
        out_path = str(tmp_path / "ensemble_bfac.pdb")
        write_ensemble_pdb(coords, aatype, res_idx, out_path, plddt_scores=plddt)
        with open(out_path) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM')]
        # B-factor is columns 60-66 in PDB format
        bfac_1 = float(lines[0][60:66])
        bfac_2 = float(lines[1][60:66])
        assert abs(bfac_1 - 85.5) < 0.1
        assert abs(bfac_2 - 42.3) < 0.1
    def test_empty_atoms_skipped(self, tmp_path):
        """Atoms with zero coordinates are not written."""
        coords = np.zeros((1, 2, ATOM_TYPE_NUM, 3), dtype=np.float32)
        # Only CA for residue 0
        coords[0, 0, CA_IDX, :] = [1, 2, 3]
        aatype = np.zeros(2, dtype=np.int32)
        res_idx = np.array([1, 2], dtype=np.int32)
        out_path = str(tmp_path / "sparse.pdb")
        write_ensemble_pdb(coords, aatype, res_idx, out_path)
        with open(out_path) as f:
            atom_lines = [l for l in f.readlines() if l.startswith('ATOM')]
        # Only 1 ATOM line (CA of residue 0), residue 1 is all-zero
        assert len(atom_lines) == 1
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
