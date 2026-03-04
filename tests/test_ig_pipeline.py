"""
tests/test_ig_pipeline.py — Unit tests for ig_pipeline.py (Tasks 1-3)
======================================================================
Covers: parse_structure_input, parse_residue_spec, generate_mask_from_structure,
        apply_ig_mask, apply_template_mask, create_template_features_from_structure,
        legacy_anchors_to_mask, compute_residue_index_with_chain_breaks.
Run:  pytest tests/test_ig_pipeline.py -v
"""
import os
import sys
import tempfile
import numpy as np
import pytest
# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ig_pipeline import (
    parse_structure_input,
    parse_residue_spec,
    generate_mask_from_structure,
    apply_ig_mask,
    apply_template_mask,
    create_template_features_from_structure,
    legacy_anchors_to_mask,
    compute_residue_index_with_chain_breaks,
    _estimate_cb_from_ca,
    _coords_array_to_atom37,
    _apply_atom_filter,
    TOKEN_DEFAULT, TOKEN_MASK, TOKEN_CENTER, TOKEN_STABLE,
    CA_IDX, CB_IDX, N_IDX, C_IDX, ATOM_TYPE_NUM,
    CHAIN_BREAK_OFFSET,
)
# ─────────────────────────────────────────────────────────────────────
# Fixtures: synthetic data generators
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def two_chain_arrays():
    """Synthetic 10-residue two-chain protein (chain A: 6 res, chain B: 4 res)."""
    n_res = 10
    sequence = "AAAAAGAAAA"  # positions 0-4=ALA, 5=GLY, 6-9=ALA
    chain_ids = np.array(['A']*6 + ['B']*4, dtype='U1')
    residue_indices = np.array([1,2,3,4,5,6, 1,2,3,4], dtype=np.int32)
    # Build CA-only coords: simple linear chain along x-axis, 3.8 Å apart
    ca_coords = np.zeros((n_res, 3), dtype=np.float32)
    ca_coords[:, 0] = np.arange(n_res) * 3.8
    return {
        'n_res': n_res,
        'sequence': sequence,
        'chain_ids': chain_ids,
        'residue_indices': residue_indices,
        'ca_coords': ca_coords,
    }
@pytest.fixture
def synthetic_pdb_file():
    """Write a minimal valid 2-chain PDB to a temp file."""
    pdb_lines = []
    atom_idx = 1
    # Chain A: 3 ALA residues
    for res_i, res_num in enumerate([1, 2, 3]):
        x = res_i * 3.8
        for atom_name, dy, dz in [('N', -0.5, 0.0), ('CA', 0.0, 0.0),
                                    ('C', 0.5, 0.0), ('CB', 0.0, 1.5), ('O', 1.0, 0.5)]:
            pdb_lines.append(
                f"ATOM  {atom_idx:>5}  {atom_name:<3} ALA A{res_num:>4}    "
                f"{x:>8.3f}{dy:>8.3f}{dz:>8.3f}  1.00  0.00           {atom_name[0]:>2}  "
            )
            atom_idx += 1
    pdb_lines.append("TER")
    # Chain B: 2 GLY residues (no CB)
    for res_i, res_num in enumerate([1, 2]):
        x = (3 + res_i) * 3.8 + 50.0  # offset to separate chains
        for atom_name, dy, dz in [('N', -0.5, 0.0), ('CA', 0.0, 0.0),
                                    ('C', 0.5, 0.0), ('O', 1.0, 0.5)]:
            pdb_lines.append(
                f"ATOM  {atom_idx:>5}  {atom_name:<3} GLY B{res_num:>4}    "
                f"{x:>8.3f}{dy:>8.3f}{dz:>8.3f}  1.00  0.00           {atom_name[0]:>2}  "
            )
            atom_idx += 1
    pdb_lines.append("TER")
    pdb_lines.append("END")
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    tmp.write('\n'.join(pdb_lines))
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)
# ─────────────────────────────────────────────────────────────────────
# Task 1: parse_structure_input
# ─────────────────────────────────────────────────────────────────────
class TestParseStructureInput:
    """Tests for the central structure parsing function."""
    def test_array_mode_ca_only_shapes(self, two_chain_arrays):
        """CA-only array input produces correct [N, 37, 3] output."""
        d = two_chain_arrays
        result = parse_structure_input(
            coords=d['ca_coords'],
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'],
            use_only_CA=True)
        assert result['all_positions'].shape == (10, ATOM_TYPE_NUM, 3)
        assert result['all_positions_mask'].shape == (10, ATOM_TYPE_NUM)
        assert result['sequence'] == d['sequence']
        assert result['chain_ids'].shape == (10,)
    def test_array_mode_ca_only_ca_present(self, two_chain_arrays):
        """CA atom slot has nonzero coords and mask=1 for all residues."""
        d = two_chain_arrays
        result = parse_structure_input(
            coords=d['ca_coords'],
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'],
            use_only_CA=True)
        # All CA masks should be 1
        assert np.all(result['all_positions_mask'][:, CA_IDX] == 1.0)
        # CA coords should match input
        np.testing.assert_allclose(
            result['all_positions'][:, CA_IDX, :], d['ca_coords'], atol=1e-5)
    def test_array_mode_ca_only_cb_estimated_for_non_gly(self, two_chain_arrays):
        """CB is estimated for non-GLY residues, absent for GLY."""
        d = two_chain_arrays
        result = parse_structure_input(
            coords=d['ca_coords'],
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'],
            use_only_CA=True)
        # GLY is at position 5 → CB mask should be 0
        assert result['all_positions_mask'][5, CB_IDX] == 0.0
        # Non-GLY (position 0) → CB mask should be 1
        assert result['all_positions_mask'][0, CB_IDX] == 1.0
        # CB should be ~1.5 Å from CA
        ca = result['all_positions'][0, CA_IDX, :]
        cb = result['all_positions'][0, CB_IDX, :]
        dist = np.linalg.norm(cb - ca)
        assert 1.0 < dist < 2.5, f"CB-CA distance {dist} outside expected range"
    def test_array_mode_ca_cb(self, two_chain_arrays):
        """CA+CB array mode places atoms correctly."""
        d = two_chain_arrays
        # Build [N, 2, 3] coords: CA at [:,0,:], CB at [:,1,:]
        ca_cb = np.zeros((10, 2, 3), dtype=np.float32)
        ca_cb[:, 0, :] = d['ca_coords']
        ca_cb[:, 1, :] = d['ca_coords'] + np.array([0, 1.5, 0])
        result = parse_structure_input(
            coords=ca_cb,
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'],
            use_CA_CB=True)
        assert np.all(result['all_positions_mask'][:, CA_IDX] == 1.0)
        # GLY (pos 5) should have CB zeroed
        assert result['all_positions_mask'][5, CB_IDX] == 0.0
        # Non-GLY should have CB
        assert result['all_positions_mask'][0, CB_IDX] == 1.0
    def test_array_mode_full_atom(self, two_chain_arrays):
        """Full-atom [N, 37, 3] array mode passes through correctly."""
        d = two_chain_arrays
        full_coords = np.zeros((10, ATOM_TYPE_NUM, 3), dtype=np.float32)
        full_coords[:, CA_IDX, :] = d['ca_coords']
        full_coords[:, N_IDX, :] = d['ca_coords'] + np.array([-0.5, 0, 0])
        result = parse_structure_input(
            coords=full_coords,
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'])
        assert result['all_positions'].shape == (10, ATOM_TYPE_NUM, 3)
        np.testing.assert_allclose(
            result['all_positions'][:, CA_IDX, :], d['ca_coords'], atol=1e-5)
    def test_pdb_mode_shapes(self, synthetic_pdb_file):
        """PDB parsing produces correct shapes (3 ALA + 2 GLY = 5 residues)."""
        result = parse_structure_input(pdb_path=synthetic_pdb_file)
        assert result['all_positions'].shape[0] == 5
        assert result['all_positions'].shape == (5, ATOM_TYPE_NUM, 3)
        assert result['sequence'] == 'AAAGG'
        assert list(result['chain_ids']) == ['A','A','A','B','B']
    def test_pdb_mode_gly_no_cb(self, synthetic_pdb_file):
        """GLY residues from PDB have CB mask = 0."""
        result = parse_structure_input(pdb_path=synthetic_pdb_file)
        # Positions 3,4 are GLY (chain B)
        assert result['all_positions_mask'][3, CB_IDX] == 0.0
        assert result['all_positions_mask'][4, CB_IDX] == 0.0
        # Position 0 is ALA → has CB
        assert result['all_positions_mask'][0, CB_IDX] == 1.0
    def test_pdb_mode_ca_only_filter(self, synthetic_pdb_file):
        """PDB with use_only_CA=True zeroes non-CA atoms."""
        result = parse_structure_input(
            pdb_path=synthetic_pdb_file, use_only_CA=True)
        # N atom should be zeroed
        assert result['all_positions_mask'][0, N_IDX] == 0.0
        # CA should still exist
        assert result['all_positions_mask'][0, CA_IDX] == 1.0
    def test_chain_break_offset(self, two_chain_arrays):
        """Chain breaks get +200 offset in residue_index."""
        d = two_chain_arrays
        result = parse_structure_input(
            coords=d['ca_coords'],
            sequence=d['sequence'],
            chain_ids=d['chain_ids'],
            residue_indices=d['residue_indices'],
            use_only_CA=True)
        res_idx = result['residue_index']
        # Chain A: original indices 1-6
        # Chain B: original indices 1-4, offset by +200
        assert res_idx[6] == 1 + CHAIN_BREAK_OFFSET
        assert res_idx[0] == 1
    def test_rejects_both_ca_and_cacb(self, two_chain_arrays):
        """Cannot set both use_only_CA and use_CA_CB."""
        d = two_chain_arrays
        with pytest.raises(AssertionError):
            parse_structure_input(
                coords=d['ca_coords'],
                sequence=d['sequence'],
                chain_ids=d['chain_ids'],
                residue_indices=d['residue_indices'],
                use_only_CA=True, use_CA_CB=True)
    def test_rejects_missing_sequence(self, two_chain_arrays):
        """Array mode without sequence raises error."""
        d = two_chain_arrays
        with pytest.raises(AssertionError):
            parse_structure_input(
                coords=d['ca_coords'],
                chain_ids=d['chain_ids'],
                residue_indices=d['residue_indices'],
                use_only_CA=True)
    def test_rejects_no_input(self):
        """No PDB or coords raises ValueError."""
        with pytest.raises(ValueError):
            parse_structure_input()
    def test_shape_mismatch_raises(self, two_chain_arrays):
        """Mismatched coords vs sequence length raises error."""
        d = two_chain_arrays
        wrong_coords = d['ca_coords'][:5]  # only 5 residues, seq has 10
        with pytest.raises(AssertionError):
            parse_structure_input(
                coords=wrong_coords,
                sequence=d['sequence'],
                chain_ids=d['chain_ids'],
                residue_indices=d['residue_indices'],
                use_only_CA=True)
# ─────────────────────────────────────────────────────────────────────
# Task 2: Residue specification parsing
# ─────────────────────────────────────────────────────────────────────
class TestParseResidueSpec:
    """Tests for the flexible residue specification parser."""
    def setup_method(self):
        """Create shared chain/residue arrays for all tests."""
        # 8 residues: chain A (res 1-5), chain B (res 1-3)
        self.chain_ids = np.array(['A']*5 + ['B']*3, dtype='U1')
        self.residue_indices = np.array([1,2,3,4,5, 1,2,3], dtype=np.int32)
        self.n_res = 8
    def test_single_chain_residue(self):
        """'B:2' selects only chain B residue 2."""
        sel = parse_residue_spec("B:2", self.chain_ids, self.residue_indices)
        assert sel.sum() == 1
        assert sel[6] == True  # B:2 is index 6 (5+1)
    def test_chain_range(self):
        """'A:2-4' selects chain A residues 2,3,4."""
        sel = parse_residue_spec("A:2-4", self.chain_ids, self.residue_indices)
        assert sel.sum() == 3
        assert list(np.where(sel)[0]) == [1, 2, 3]
    def test_entire_chain(self):
        """'B:*' selects all chain B residues."""
        sel = parse_residue_spec("B:*", self.chain_ids, self.residue_indices)
        assert sel.sum() == 3
        assert list(np.where(sel)[0]) == [5, 6, 7]
    def test_comma_separated_chain_specs(self):
        """'A:1,A:3,B:1' selects three specific residues."""
        sel = parse_residue_spec("A:1,A:3,B:1", self.chain_ids, self.residue_indices)
        assert sel.sum() == 3
        assert list(np.where(sel)[0]) == [0, 2, 5]
    def test_raw_indices(self):
        """'0,7' selects 0-indexed positions directly."""
        sel = parse_residue_spec("0,7", self.chain_ids, self.residue_indices)
        assert sel.sum() == 2
        assert sel[0] == True
        assert sel[7] == True
    def test_mixed_format(self):
        """'A:1,6' mixes chain spec and raw index."""
        sel = parse_residue_spec("A:1,6", self.chain_ids, self.residue_indices)
        assert sel[0] == True   # A:1
        assert sel[6] == True   # raw index 6
    def test_out_of_range_index_ignored(self):
        """Raw index beyond N_res is silently ignored."""
        sel = parse_residue_spec("99", self.chain_ids, self.residue_indices)
        assert sel.sum() == 0
    def test_nonexistent_chain_selects_nothing(self):
        """'C:1' on a protein with only chains A,B selects nothing."""
        sel = parse_residue_spec("C:1", self.chain_ids, self.residue_indices)
        assert sel.sum() == 0
# ─────────────────────────────────────────────────────────────────────
# Task 2: Mask generation
# ─────────────────────────────────────────────────────────────────────
class TestGenerateMask:
    """Tests for generate_mask_from_structure."""
    def _make_positions(self, n_res):
        """Build dummy all_positions with CA along x-axis, 3.8 Å apart."""
        pos = np.zeros((n_res, ATOM_TYPE_NUM, 3), dtype=np.float32)
        pos[:, CA_IDX, 0] = np.arange(n_res) * 3.8
        return pos
    def test_mask_residues_only(self):
        """Only --mask_residues sets token -1, rest stays 0."""
        chain_ids = np.array(['A']*5 + ['B']*3, dtype='U1')
        res_idx = np.array([1,2,3,4,5, 1,2,3], dtype=np.int32)
        pos = self._make_positions(8)
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx, mask_residues="B:2")
        assert mask[6] == TOKEN_MASK     # B:2
        assert mask[0] == TOKEN_DEFAULT  # A:1 untouched
        assert np.sum(mask == TOKEN_MASK) == 1
    def test_auto_sampling_radius(self):
        """Auto-radius creates -2 centers around masked residues."""
        n_res = 10
        chain_ids = np.array(['A']*n_res, dtype='U1')
        res_idx = np.arange(1, n_res + 1, dtype=np.int32)
        pos = self._make_positions(n_res)  # 3.8 Å spacing
        # Mask residue 5 (index 4), radius 5.0 Å → should catch neighbors at 3.8 Å
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx,
            mask_residues="A:5", auto_sampling_radius=5.0)
        assert mask[4] == TOKEN_MASK   # A:5 → masked
        assert mask[3] == TOKEN_CENTER # A:4 → within 3.8 Å → auto center
        assert mask[5] == TOKEN_CENTER # A:6 → within 3.8 Å → auto center
        # A:1 (index 0) is 4*3.8=15.2 Å away → should be default
        assert mask[0] == TOKEN_DEFAULT
    def test_stable_overrides_auto_center(self):
        """Stable token (+1) overrides auto-assigned sampling center (-2)."""
        n_res = 10
        chain_ids = np.array(['A']*n_res, dtype='U1')
        res_idx = np.arange(1, n_res + 1, dtype=np.int32)
        pos = self._make_positions(n_res)
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx,
            mask_residues="A:5",
            auto_sampling_radius=5.0,
            stable_residues="A:4")  # neighbor would be -2 but stable wins
        assert mask[4] == TOKEN_MASK    # A:5 itself
        assert mask[3] == TOKEN_STABLE  # A:4 → stable overrides auto -2
        assert mask[5] == TOKEN_CENTER  # A:6 → still auto center
    def test_explicit_sampling_centers(self):
        """Explicit --sampling_centers sets -2 directly."""
        chain_ids = np.array(['A']*5, dtype='U1')
        res_idx = np.array([1,2,3,4,5], dtype=np.int32)
        pos = self._make_positions(5)
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx,
            mask_residues="A:3", sampling_centers="A:1,A:5")
        assert mask[2] == TOKEN_MASK    # A:3
        assert mask[0] == TOKEN_CENTER  # A:1
        assert mask[4] == TOKEN_CENTER  # A:5
        assert mask[1] == TOKEN_DEFAULT # A:2 untouched
    def test_explicit_centers_dont_override_mask(self):
        """Specifying a residue as both mask and center keeps it as mask (-1)."""
        chain_ids = np.array(['A']*5, dtype='U1')
        res_idx = np.array([1,2,3,4,5], dtype=np.int32)
        pos = self._make_positions(5)
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx,
            mask_residues="A:3", sampling_centers="A:3")
        assert mask[2] == TOKEN_MASK  # mask wins over center
    def test_load_from_npy(self, tmp_path):
        """Loading mask from .npy file works correctly."""
        expected = np.array([0, -1, -2, 1, 0], dtype=np.int32)
        npy_path = str(tmp_path / "mask.npy")
        np.save(npy_path, expected)
        chain_ids = np.array(['A']*5, dtype='U1')
        res_idx = np.array([1,2,3,4,5], dtype=np.int32)
        pos = self._make_positions(5)
        mask = generate_mask_from_structure(
            pos, chain_ids, res_idx, mask_array_path=npy_path)
        np.testing.assert_array_equal(mask, expected)
    def test_no_args_all_default(self):
        """No mask args → all tokens are 0 (default)."""
        chain_ids = np.array(['A']*5, dtype='U1')
        res_idx = np.array([1,2,3,4,5], dtype=np.int32)
        pos = self._make_positions(5)
        mask = generate_mask_from_structure(pos, chain_ids, res_idx)
        assert np.all(mask == TOKEN_DEFAULT)
# ─────────────────────────────────────────────────────────────────────
# Task 2: apply_ig_mask
# ─────────────────────────────────────────────────────────────────────
class TestApplyIgMask:
    """Tests for vectorized IG coordinate masking."""
    def test_mask_token_zeros_coords(self):
        """Token -1 residues get all coordinates zeroed."""
        pos = np.random.randn(5, ATOM_TYPE_NUM, 3).astype(np.float32)
        mask_arr = np.array([0, -1, 0, -1, 1], dtype=np.int32)
        masked = apply_ig_mask(pos, mask_arr)
        # Masked residues should be all zero
        np.testing.assert_array_equal(masked[1], 0.0)
        np.testing.assert_array_equal(masked[3], 0.0)
        # Non-masked residues should be unchanged
        np.testing.assert_array_equal(masked[0], pos[0])
        np.testing.assert_array_equal(masked[2], pos[2])
        np.testing.assert_array_equal(masked[4], pos[4])
    def test_does_not_modify_original(self):
        """apply_ig_mask returns a copy, not a view."""
        pos = np.ones((3, ATOM_TYPE_NUM, 3), dtype=np.float32)
        mask_arr = np.array([-1, 0, 0], dtype=np.int32)
        masked = apply_ig_mask(pos, mask_arr)
        assert np.all(pos[0] == 1.0)  # original unchanged
        assert np.all(masked[0] == 0.0)  # copy is zeroed
    def test_center_and_stable_keep_coords(self):
        """Tokens -2 and +1 preserve coordinates."""
        pos = np.random.randn(4, ATOM_TYPE_NUM, 3).astype(np.float32)
        mask_arr = np.array([-2, 1, 0, -1], dtype=np.int32)
        masked = apply_ig_mask(pos, mask_arr)
        np.testing.assert_array_equal(masked[0], pos[0])  # center
        np.testing.assert_array_equal(masked[1], pos[1])  # stable
        np.testing.assert_array_equal(masked[2], pos[2])  # default
# ─────────────────────────────────────────────────────────────────────
# Task 2: Legacy backward compatibility
# ─────────────────────────────────────────────────────────────────────
class TestLegacyAnchorsToMask:
    """Tests for legacy pMHC anchor-to-mask conversion."""
    def test_basic_pmhc_conversion(self):
        """MHC(180 res) + peptide(9 res), anchors at 2,9 → correct mask."""
        n_res = 189
        pep_len = 9
        anchors = [2, 9]  # 1-indexed within peptide
        mask = legacy_anchors_to_mask(n_res, pep_len, anchors)
        mhc_len = 180
        # MHC region: all default
        assert np.all(mask[:mhc_len] == TOKEN_DEFAULT)
        # Anchor positions (0-indexed absolute): 181, 188
        assert mask[181] == TOKEN_DEFAULT  # anchor 2 → kept
        assert mask[188] == TOKEN_DEFAULT  # anchor 9 → kept
        # Non-anchor peptide: masked
        assert mask[182] == TOKEN_MASK
        assert mask[183] == TOKEN_MASK
        assert mask[187] == TOKEN_MASK
    def test_all_anchors_no_masking(self):
        """If every peptide position is an anchor, nothing gets masked."""
        mask = legacy_anchors_to_mask(13, 3, [1, 2, 3])
        assert np.all(mask == TOKEN_DEFAULT)
# ─────────────────────────────────────────────────────────────────────
# Task 3: Template masking
# ─────────────────────────────────────────────────────────────────────
class TestApplyTemplateMask:
    """Tests for evoformer template feature masking."""
    def test_mask_zeros_template_positions(self):
        """Token -1 residues get all template features zeroed."""
        n_tmpl, n_res = 2, 5
        tf = {
            'template_all_atom_positions': np.random.randn(n_tmpl, n_res, ATOM_TYPE_NUM, 3).astype(np.float32),
            'template_all_atom_masks': np.ones((n_tmpl, n_res, ATOM_TYPE_NUM), dtype=np.float32),
            'template_pseudo_beta': np.random.randn(n_tmpl, n_res, 3).astype(np.float32),
            'template_pseudo_beta_mask': np.ones((n_tmpl, n_res), dtype=np.float32),
        }
        evo_mask = np.array([0, -1, 0, 0, -1], dtype=np.int32)
        result = apply_template_mask(tf, evo_mask)
        # Masked residues (1, 4) should be zero in all templates
        for t in range(n_tmpl):
            np.testing.assert_array_equal(result['template_all_atom_positions'][t, 1], 0.0)
            np.testing.assert_array_equal(result['template_all_atom_positions'][t, 4], 0.0)
            assert result['template_all_atom_masks'][t, 1].sum() == 0.0
            assert result['template_pseudo_beta_mask'][t, 1] == 0.0
        # Non-masked residues should be unchanged
        assert result['template_all_atom_masks'][0, 0].sum() > 0
        assert result['template_pseudo_beta_mask'][0, 0] == 1.0
    def test_no_mask_tokens_no_change(self):
        """All-default mask leaves template features unchanged."""
        n_tmpl, n_res = 1, 3
        original_pos = np.random.randn(n_tmpl, n_res, ATOM_TYPE_NUM, 3).astype(np.float32)
        tf = {
            'template_all_atom_positions': original_pos.copy(),
            'template_all_atom_masks': np.ones((n_tmpl, n_res, ATOM_TYPE_NUM), dtype=np.float32),
        }
        evo_mask = np.array([0, 0, 0], dtype=np.int32)
        apply_template_mask(tf, evo_mask)
        np.testing.assert_array_equal(tf['template_all_atom_positions'], original_pos)
# ─────────────────────────────────────────────────────────────────────
# Task 1: Template feature builder from structure
# ─────────────────────────────────────────────────────────────────────
class TestCreateTemplateFeatures:
    """Tests for create_template_features_from_structure."""
    def test_output_shapes(self):
        """Template features have correct shapes and batch dim [1, ...]."""
        n_res = 5
        pos = np.random.randn(n_res, ATOM_TYPE_NUM, 3).astype(np.float32)
        mask = np.zeros((n_res, ATOM_TYPE_NUM), dtype=np.float32)
        mask[:, CA_IDX] = 1.0
        mask[:, CB_IDX] = 1.0
        tf = create_template_features_from_structure(pos, mask, "AAAAA")
        assert tf['template_all_atom_positions'].shape == (1, n_res, ATOM_TYPE_NUM, 3)
        assert tf['template_all_atom_masks'].shape == (1, n_res, ATOM_TYPE_NUM)
        assert tf['template_aatype'].shape == (1, n_res, 22)
        assert tf['template_pseudo_beta'].shape == (1, n_res, 3)
        assert tf['template_pseudo_beta_mask'].shape == (1, n_res)
    def test_gly_pseudo_beta_uses_ca(self):
        """GLY residues use CA for pseudo-beta, non-GLY use CB."""
        n_res = 3
        pos = np.zeros((n_res, ATOM_TYPE_NUM, 3), dtype=np.float32)
        pos[:, CA_IDX, :] = [[1,0,0], [2,0,0], [3,0,0]]
        pos[:, CB_IDX, :] = [[1,1,0], [2,1,0], [3,1,0]]
        mask = np.zeros((n_res, ATOM_TYPE_NUM), dtype=np.float32)
        mask[:, CA_IDX] = 1.0
        mask[:, CB_IDX] = 1.0
        mask[1, CB_IDX] = 0.0  # GLY has no CB
        # Sequence: AGA (middle is GLY)
        tf = create_template_features_from_structure(pos, mask, "AGA")
        pb = tf['template_pseudo_beta'][0]  # [3, 3]
        # ALA (pos 0): pseudo_beta = CB = [1,1,0]
        np.testing.assert_allclose(pb[0], [1, 1, 0])
        # GLY (pos 1): pseudo_beta = CA = [2,0,0]
        np.testing.assert_allclose(pb[1], [2, 0, 0])
    def test_missing_cb_falls_back_to_ca(self):
        """Non-GLY with missing CB falls back to CA for pseudo-beta."""
        pos = np.zeros((1, ATOM_TYPE_NUM, 3), dtype=np.float32)
        pos[0, CA_IDX, :] = [5, 5, 5]
        # CB is zero/missing
        mask = np.zeros((1, ATOM_TYPE_NUM), dtype=np.float32)
        mask[0, CA_IDX] = 1.0
        # ALA but no CB atom
        tf = create_template_features_from_structure(pos, mask, "A")
        np.testing.assert_allclose(tf['template_pseudo_beta'][0, 0], [5, 5, 5])
# ─────────────────────────────────────────────────────────────────────
# Chain break computation
# ─────────────────────────────────────────────────────────────────────
class TestChainBreakOffset:
    """Tests for residue_index chain break computation."""
    def test_single_chain_no_offset(self):
        """Single chain has no offset applied."""
        chain_ids = np.array(['A', 'A', 'A'], dtype='U1')
        res_idx = np.array([1, 2, 3], dtype=np.int32)
        result = compute_residue_index_with_chain_breaks(chain_ids, res_idx)
        np.testing.assert_array_equal(result, [1, 2, 3])
    def test_two_chains_one_break(self):
        """Two chains get one +200 offset."""
        chain_ids = np.array(['A', 'A', 'B', 'B'], dtype='U1')
        res_idx = np.array([1, 2, 1, 2], dtype=np.int32)
        result = compute_residue_index_with_chain_breaks(chain_ids, res_idx)
        np.testing.assert_array_equal(result, [1, 2, 201, 202])
    def test_three_chains_two_breaks(self):
        """Three chains get two cumulative +200 offsets."""
        chain_ids = np.array(['A', 'B', 'C'], dtype='U1')
        res_idx = np.array([1, 1, 1], dtype=np.int32)
        result = compute_residue_index_with_chain_breaks(chain_ids, res_idx)
        np.testing.assert_array_equal(result, [1, 201, 401])
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
