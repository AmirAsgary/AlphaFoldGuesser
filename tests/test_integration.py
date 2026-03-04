"""
tests/test_integration.py — End-to-end integration tests for the IG pipeline
==============================================================================
Tests the full flow: parse input → generate mask → apply mask → build IG
features → build template features. Does NOT require AlphaFold model weights
or JAX model inference — only tests the data preparation pipeline.
Run:  pytest tests/test_integration.py -v
"""
import os
import sys
import tempfile
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ig_pipeline import (
    parse_structure_input,
    generate_mask_from_structure,
    apply_ig_mask,
    apply_template_mask,
    create_template_features_from_structure,
    initial_guess_features_v2,
    legacy_anchors_to_mask,
    TOKEN_DEFAULT, TOKEN_MASK, TOKEN_CENTER, TOKEN_STABLE,
    CA_IDX, CB_IDX, ATOM_TYPE_NUM,
)
from sampling_utils import compute_residue_sets
# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def multichain_pdb(tmp_path):
    """Write a 3-chain PDB: chain A (5 ALA), chain B (3 ALA), chain C (2 GLY).
    Chains are placed along x-axis with large gaps between chains."""
    lines = []
    atom_idx = 1
    chains = [
        ('A', 'ALA', 5, 0.0),
        ('B', 'ALA', 3, 100.0),
        ('C', 'GLY', 2, 200.0),
    ]
    for chain_id, resname, n_res, x_offset in chains:
        atoms_for_res = [('N', -0.5, 0.0), ('CA', 0.0, 0.0), ('C', 0.5, 0.0), ('O', 1.0, 0.5)]
        if resname == 'ALA':
            atoms_for_res.append(('CB', 0.0, 1.5))
        for ri in range(n_res):
            x = x_offset + ri * 3.8
            for aname, dy, dz in atoms_for_res:
                lines.append(
                    f"ATOM  {atom_idx:>5}  {aname:<3} {resname} {chain_id}{ri+1:>4}    "
                    f"{x:>8.3f}{dy:>8.3f}{dz:>8.3f}  1.00  0.00           {aname[0]:>2}  "
                )
                atom_idx += 1
        lines.append("TER")
    lines.append("END")
    pdb_path = str(tmp_path / "multichain.pdb")
    with open(pdb_path, 'w') as f:
        f.write('\n'.join(lines))
    return pdb_path
# ─────────────────────────────────────────────────────────────────────
# Full pipeline: PDB → parse → mask → IG features
# ─────────────────────────────────────────────────────────────────────
class TestFullPipelinePDB:
    """End-to-end test: PDB input through to IG and template features."""
    def test_parse_mask_apply_ig(self, multichain_pdb):
        """Parse PDB → generate mask → apply IG mask → verify zeroing."""
        # Step 1: Parse
        struct = parse_structure_input(pdb_path=multichain_pdb)
        assert struct['all_positions'].shape[0] == 10  # 5+3+2
        assert struct['sequence'] == 'AAAAAAAAGG'
        assert list(np.unique(struct['chain_ids'])) == ['A', 'B', 'C']
        # Step 2: Generate mask — mask chain B residue 2, auto-radius around it
        mask = generate_mask_from_structure(
            struct['all_positions'],
            struct['chain_ids'],
            struct['residue_indices'],
            mask_residues="B:2",
            auto_sampling_radius=5.0,
            stable_residues="A:*")
        # Chain A (5 res) should all be stable
        assert np.all(mask[:5] == TOKEN_STABLE)
        # B:2 is index 6 (5 chain-A + 1 for B:1 + 0-index) → actually index 6
        # Chain B indices: 5=B:1, 6=B:2, 7=B:3
        assert mask[6] == TOKEN_MASK  # B:2 masked
        # B:1 and B:3 are at 3.8 Å from B:2 → within radius=5 → center
        # But they are NOT stable (only A is stable), so they become -2
        assert mask[5] == TOKEN_CENTER  # B:1
        assert mask[7] == TOKEN_CENTER  # B:3
        # Chain C is far away (200 Å offset) → stays default
        assert mask[8] == TOKEN_DEFAULT
        assert mask[9] == TOKEN_DEFAULT
        # Step 3: Apply IG mask
        masked_pos = apply_ig_mask(struct['all_positions'], mask)
        # B:2 (index 6) should be all zeros
        assert np.all(masked_pos[6] == 0.0)
        # A:1 (index 0) should be unchanged (stable)
        np.testing.assert_array_equal(
            masked_pos[0], struct['all_positions'][0])
        # B:1 (index 5, center) should keep its coords
        np.testing.assert_array_equal(
            masked_pos[5], struct['all_positions'][5])
    def test_template_features_from_pdb(self, multichain_pdb):
        """Template features built from PDB have correct shapes and content."""
        struct = parse_structure_input(pdb_path=multichain_pdb)
        tf = create_template_features_from_structure(
            struct['all_positions'],
            struct['all_positions_mask'],
            struct['sequence'])
        n_res = 10
        assert tf['template_all_atom_positions'].shape == (1, n_res, ATOM_TYPE_NUM, 3)
        assert tf['template_aatype'].shape == (1, n_res, 22)
        assert tf['template_pseudo_beta'].shape == (1, n_res, 3)
        # GLY residues (positions 8,9) should use CA for pseudo-beta
        pb = tf['template_pseudo_beta'][0]
        ca = struct['all_positions'][:, CA_IDX, :]
        np.testing.assert_allclose(pb[8], ca[8], atol=1e-5)
        np.testing.assert_allclose(pb[9], ca[9], atol=1e-5)
    def test_template_masking_combined_with_ig(self, multichain_pdb):
        """IG mask + template mask (auto) zeros both paths consistently."""
        struct = parse_structure_input(pdb_path=multichain_pdb)
        mask = generate_mask_from_structure(
            struct['all_positions'],
            struct['chain_ids'],
            struct['residue_indices'],
            mask_residues="B:2")
        # Apply IG mask
        masked_pos = apply_ig_mask(struct['all_positions'], mask)
        assert np.all(masked_pos[6] == 0.0)
        # Build template features and apply same mask (auto mode)
        tf = create_template_features_from_structure(
            struct['all_positions'],
            struct['all_positions_mask'],
            struct['sequence'])
        apply_template_mask(tf, mask)
        # Template for B:2 (index 6) should also be zeroed
        assert np.all(tf['template_all_atom_positions'][0, 6] == 0.0)
        assert tf['template_all_atom_masks'][0, 6].sum() == 0.0
        assert tf['template_pseudo_beta_mask'][0, 6] == 0.0
        # A:1 (index 0) should be unaffected in template
        assert tf['template_all_atom_masks'][0, 0].sum() > 0
# ─────────────────────────────────────────────────────────────────────
# Full pipeline: Numpy arrays → parse → mask → residue sets
# ─────────────────────────────────────────────────────────────────────
class TestFullPipelineArrays:
    """End-to-end test: array input through to sampling residue sets."""
    def test_ca_only_through_to_residue_sets(self):
        """CA-only array → parse → mask → residue sets → verify geometry."""
        n_res = 15
        sequence = "A" * n_res
        chain_ids = np.array(['A']*10 + ['B']*5, dtype='U1')
        residue_indices = np.concatenate([
            np.arange(1, 11), np.arange(1, 6)]).astype(np.int32)
        # Place CA atoms along x-axis
        ca = np.zeros((n_res, 3), dtype=np.float32)
        ca[:, 0] = np.arange(n_res) * 3.8
        # Parse
        struct = parse_structure_input(
            coords=ca, sequence=sequence,
            chain_ids=chain_ids, residue_indices=residue_indices,
            use_only_CA=True)
        assert struct['all_positions'].shape == (n_res, ATOM_TYPE_NUM, 3)
        # Mask: B:3 (index 12), auto-radius 8 Å, stable A:*
        mask = generate_mask_from_structure(
            struct['all_positions'], chain_ids, residue_indices,
            mask_residues="B:3",
            auto_sampling_radius=8.0,
            stable_residues="A:*")
        # B:3 is at index 12
        assert mask[12] == TOKEN_MASK
        # Chain A all stable
        assert np.all(mask[:10] == TOKEN_STABLE)
        # B:1 (idx 10) is 2*3.8=7.6 Å from B:3 → within 8 Å radius
        assert mask[10] == TOKEN_CENTER
        # B:2 (idx 11) is 1*3.8=3.8 Å → within radius
        assert mask[11] == TOKEN_CENTER
        # B:4 (idx 13) is 1*3.8=3.8 Å → within radius
        assert mask[13] == TOKEN_CENTER
        # B:5 (idx 14) is 2*3.8=7.6 Å → within radius
        assert mask[14] == TOKEN_CENTER
        # Compute residue sets
        sets = compute_residue_sets(mask, struct['all_positions'], radius=8.0)
        # Mask set: only index 12
        assert sets['mask_set'].sum() == 1
        assert sets['mask_set'][12] == True
        # Stable: indices 0-9
        assert sets['stable_set'].sum() == 10
        # Centers: B:1,2,4,5 = indices 10,11,13,14
        assert sets['centers'].sum() == 4
        # Sampleable: centers + eligible (no eligible here since
        # only stable and centers, no defaults within radius)
        assert sets['sampleable'].sum() == 4
    def test_ig_features_v2_produces_jax_array(self):
        """initial_guess_features_v2 returns a JAX array of correct shape."""
        pytest.importorskip("jax")
        import jax.numpy as jnp
        n_res = 5
        struct = parse_structure_input(
            coords=np.random.randn(n_res, 3).astype(np.float32),
            sequence="AAAAA",
            chain_ids=np.array(['A']*n_res, dtype='U1'),
            residue_indices=np.arange(1, n_res+1, dtype=np.int32),
            use_only_CA=True)
        mask = np.array([0, -1, 0, 0, 0], dtype=np.int32)
        ig = initial_guess_features_v2(struct, mask_array=mask)
        
        assert ig.shape == (n_res, ATOM_TYPE_NUM, 3)
        assert isinstance(ig, jnp.ndarray)
        
        # FIX: Convert JAX array to standard numpy array before indexing
        ig_np = np.asarray(ig)
        
        # Masked residue (index 1) should be all zeros
        assert float(np.abs(ig_np[1]).sum()) == 0.0
        # Non-masked residues should have nonzero CA
        assert float(np.abs(ig_np[0, CA_IDX]).sum()) > 0.0
# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    """Edge case tests for robustness."""
    def test_single_residue_protein(self):
        """Pipeline handles a single-residue protein without error."""
        struct = parse_structure_input(
            coords=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            sequence="A",
            chain_ids=np.array(['A'], dtype='U1'),
            residue_indices=np.array([1], dtype=np.int32),
            use_only_CA=True)
        assert struct['all_positions'].shape == (1, ATOM_TYPE_NUM, 3)
        mask = generate_mask_from_structure(
            struct['all_positions'],
            struct['chain_ids'],
            struct['residue_indices'])
        assert mask.shape == (1,)
        assert mask[0] == TOKEN_DEFAULT
    def test_all_gly_protein(self):
        """All-glycine protein works (no CB anywhere)."""
        n_res = 5
        struct = parse_structure_input(
            coords=np.random.randn(n_res, 3).astype(np.float32),
            sequence="G" * n_res,
            chain_ids=np.array(['A']*n_res, dtype='U1'),
            residue_indices=np.arange(1, n_res+1, dtype=np.int32),
            use_only_CA=True)
        # GLY should have CB mask = 0 for all residues
        assert np.all(struct['all_positions_mask'][:, CB_IDX] == 0.0)
        # CA should still exist
        assert np.all(struct['all_positions_mask'][:, CA_IDX] == 1.0)
        # Template features should use CA for pseudo-beta
        tf = create_template_features_from_structure(
            struct['all_positions'],
            struct['all_positions_mask'],
            struct['sequence'])
        ca = struct['all_positions'][:, CA_IDX, :]
        np.testing.assert_allclose(tf['template_pseudo_beta'][0], ca, atol=1e-5)
    def test_mask_all_residues(self):
        """Masking every residue produces all-zero IG coordinates."""
        n_res = 4
        struct = parse_structure_input(
            coords=np.ones((n_res, 3), dtype=np.float32) * 10.0,
            sequence="AAAA",
            chain_ids=np.array(['A']*n_res, dtype='U1'),
            residue_indices=np.arange(1, n_res+1, dtype=np.int32),
            use_only_CA=True)
        mask = np.full(n_res, TOKEN_MASK, dtype=np.int32)
        masked = apply_ig_mask(struct['all_positions'], mask)
        assert np.all(masked == 0.0)
    def test_mask_no_residues_preserves_all(self):
        """Empty mask preserves all coordinates identically."""
        n_res = 4
        struct = parse_structure_input(
            coords=np.random.randn(n_res, 3).astype(np.float32),
            sequence="AAAA",
            chain_ids=np.array(['A']*n_res, dtype='U1'),
            residue_indices=np.arange(1, n_res+1, dtype=np.int32),
            use_only_CA=True)
        mask = np.zeros(n_res, dtype=np.int32)
        masked = apply_ig_mask(struct['all_positions'], mask)
        np.testing.assert_array_equal(masked, struct['all_positions'])
    def test_legacy_pmhc_round_trip(self):
        """Legacy anchor mask matches expected pMHC behavior."""
        # Simulated: 180 MHC + 9 peptide, anchors at positions 2 and 9
        n_res = 189
        mhc_len = 180
        pep_len = 9
        anchors = [2, 9]
        mask = legacy_anchors_to_mask(n_res, pep_len, anchors, mhc_len)
        # MHC: all default
        assert np.all(mask[:mhc_len] == TOKEN_DEFAULT)
        # Anchor absolute positions: 181 (2-1+180), 188 (9-1+180)
        assert mask[181] == TOKEN_DEFAULT
        assert mask[188] == TOKEN_DEFAULT
        # Non-anchor peptide: 180,182,183,184,185,186,187 → masked
        non_anchor_peptide = [i for i in range(mhc_len, n_res)
                              if i not in (181, 188)]
        for idx in non_anchor_peptide:
            assert mask[idx] == TOKEN_MASK, f"Position {idx} should be masked"
        # Total masked = 9 - 2 = 7
        assert np.sum(mask == TOKEN_MASK) == 7
    def test_sampling_sets_with_no_mask_residues(self):
        """Residue sets with only centers (no mask tokens) work correctly."""
        n_res = 5
        mask = np.array([0, 0, -2, 0, 0], dtype=np.int32)
        pos = np.zeros((n_res, ATOM_TYPE_NUM, 3), dtype=np.float32)
        pos[:, CA_IDX, 0] = np.arange(n_res) * 3.8
        sets = compute_residue_sets(mask, pos, radius=5.0)
        assert sets['mask_set'].sum() == 0
        assert sets['centers'][2] == True
        # Neighbors within 5 Å: positions 1 (3.8 Å) and 3 (3.8 Å)
        assert sets['eligible'][1] == True
        assert sets['eligible'][3] == True
# ─────────────────────────────────────────────────────────────────────
# Conftest: create tests/__init__.py if needed
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True, scope='session')
def ensure_tests_package(tmp_path_factory):
    """Ensure tests/ has an __init__.py for import resolution."""
    tests_dir = os.path.dirname(__file__)
    init_path = os.path.join(tests_dir, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('')
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
