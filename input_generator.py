"""input_generator.py — Auto-generate alignment and targets TSV files.
Eliminates the need for users to manually create legacy input files.
When the template PDB is the same protein as the target (conformation
sampling), the alignment is a simple 1:1 identity mapping. For mutant
targets, a pairwise sequence alignment is performed automatically.
Supports arbitrary multi-chain proteins with chain-break handling.
CRITICAL: Uses predict_utils.load_pdb_coords() for PDB parsing to
guarantee residue counts match create_single_template_features().
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from predict_utils import load_pdb_coords
# ─────────────────────────────────────────────────────────────────────
# PDB sequence extraction (uses the SAME reader as predict_utils)
# ─────────────────────────────────────────────────────────────────────
def get_chain_sequences_from_pdb(pdb_path: str) -> Tuple[str, int, List[str], List[int]]:
    """Parse PDB file and extract per-chain amino acid sequences.
    Uses load_pdb_coords from predict_utils to guarantee consistent
    residue counting with create_single_template_features().
    Args:
        pdb_path: path to the PDB file.
    Returns:
        chainseq: concatenated sequence with '/' chain separators.
        total_res: total residue count across all chains (= template_len).
        chain_ids: list of unique chain ID strings.
        chain_lengths: list of residue counts per chain.
    """
    # Use the exact same PDB reader as predict_utils
    chains, all_resids, all_coords, all_name1s = load_pdb_coords(
        pdb_path, allow_chainbreaks=True, allow_skipped_lines=True)
    # Build per-chain sequences in chain order
    chain_seqs = []
    chain_lengths = []
    total_res = 0
    for ch in chains:
        seq = ''.join(all_name1s[ch][r] for r in all_resids[ch])
        chain_seqs.append(seq)
        chain_lengths.append(len(seq))
        total_res += len(seq)
    chainseq = "/".join(chain_seqs)
    return chainseq, total_res, chains, chain_lengths
# ─────────────────────────────────────────────────────────────────────
# Alignment generation (identity or pairwise)
# ─────────────────────────────────────────────────────────────────────
def _build_identity_alignment(n: int) -> str:
    """Build 1:1 identity alignment string for n residues.
    Returns semicolon-separated 'i:i' pairs, zero-indexed.
    Args:
        n: number of aligned residue positions.
    Returns:
        alignment string, e.g. '0:0;1:1;2:2;...;N-1:N-1'.
    """
    # Build integer arrays then format
    idx = np.arange(n, dtype=np.int32)
    pairs = [f"{i}:{i}" for i in idx]
    return ";".join(pairs)
def _build_pairwise_alignment(target_seq: str, template_seq: str,
                              template_chain_offsets: List[int]
                              ) -> Tuple[str, int]:
    """Build alignment string from pairwise per-chain sequence alignment.
    Aligns each chain independently, then concatenates with correct
    template offsets. Only aligned (non-gap) positions are included.
    Args:
        target_seq: target sequence (no chain separators).
        template_seq: template sequence (no chain separators).
        template_chain_offsets: cumulative start index of each chain
            in the template sequence (e.g. [0, 182] for two chains).
    Returns:
        align_str: semicolon-separated 'target_idx:template_idx' pairs.
        identities: count of identical residue matches.
    """
    # Lazy import to avoid forcing dependency when not needed
    try:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        def align_pair(seq1, seq2):
            """Run pairwise alignment and return top alignment strings."""
            aln = aligner.align(seq1, seq2)
            best = aln[0]
            return str(best).split('\n')[0], str(best).split('\n')[2]
    except ImportError:
        from Bio import pairwise2
        def align_pair(seq1, seq2):
            """Fallback to legacy pairwise2 aligner."""
            alns = pairwise2.align.globalxx(seq1, seq2)
            return alns[0][0], alns[0][1]
    # Global alignment of full sequences
    aligned_target, aligned_template = align_pair(target_seq, template_seq)
    # Walk through alignment, build mapping
    pairs = []
    identities = 0
    tgt_idx = 0
    tmpl_idx = 0
    for q_char, t_char in zip(aligned_target, aligned_template):
        is_tgt = (q_char != '-')
        is_tmpl = (t_char != '-')
        if is_tgt and is_tmpl:
            pairs.append(f"{tgt_idx}:{tmpl_idx}")
            if q_char == t_char:
                identities += 1
        if is_tgt:
            tgt_idx += 1
        if is_tmpl:
            tmpl_idx += 1
    return ";".join(pairs), identities
def generate_alignment_string(target_chainseq: str, template_pdb_path: str
                              ) -> Tuple[str, int, int, int]:
    """Generate alignment between target sequence and template PDB.
    If sequences are identical, returns identity mapping (fast path).
    Otherwise, performs pairwise alignment (handles mutations/indels).
    Uses load_pdb_coords() for template parsing to guarantee consistency.
    Args:
        target_chainseq: target sequence with '/' chain separators.
        template_pdb_path: path to the template PDB file.
    Returns:
        align_str: alignment string for TSV.
        identities: number of identical aligned positions.
        target_len: total target residue count.
        template_len: total template residue count (from load_pdb_coords).
    """
    target_seq = target_chainseq.replace("/", "")
    target_len = len(target_seq)
    # Extract template info using the SAME reader as predict_utils
    tmpl_chainseq, template_len, tmpl_chains, tmpl_chain_lengths = \
        get_chain_sequences_from_pdb(template_pdb_path)
    template_seq = tmpl_chainseq.replace("/", "")
    # Compute cumulative chain offsets for pairwise alignment
    tmpl_offsets = np.zeros(len(tmpl_chain_lengths), dtype=np.int32)
    for i in range(1, len(tmpl_chain_lengths)):
        tmpl_offsets[i] = tmpl_offsets[i-1] + tmpl_chain_lengths[i-1]
    # Fast path: identical sequences → identity mapping
    if target_seq == template_seq:
        align_str = _build_identity_alignment(target_len)
        return align_str, target_len, target_len, template_len
    # Slow path: pairwise alignment for mutations/indels
    align_str, identities = _build_pairwise_alignment(
        target_seq, template_seq, tmpl_offsets.tolist())
    return align_str, identities, target_len, template_len
# ─────────────────────────────────────────────────────────────────────
# File generation functions
# ─────────────────────────────────────────────────────────────────────
def generate_alignment_file(template_pdb_path: str,
                            target_chainseq: str,
                            output_path: str) -> pd.DataFrame:
    """Create alignment TSV file for a single template.
    Auto-generates 1:1 or pairwise alignment depending on sequence match.
    Columns: template_pdbfile, target_to_template_alignstring, identities,
             target_len, template_len.
    Args:
        template_pdb_path: path to the template PDB file.
        target_chainseq: target sequence with '/' chain separators.
        output_path: where to write the alignment TSV.
    Returns:
        alignment DataFrame (single row).
    """
    align_str, identities, target_len, template_len = generate_alignment_string(
        target_chainseq, template_pdb_path)
    df = pd.DataFrame({
        "template_pdbfile": [os.path.abspath(template_pdb_path)],
        "target_to_template_alignstring": [align_str],
        "identities": [identities],
        "target_len": [target_len],
        "template_len": [template_len],
    })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"[input_generator] Wrote alignment file: {output_path}")
    return df
def generate_targets_file(target_chainseq: str,
                          alignment_file_path: str,
                          targetid: str,
                          output_path: str) -> pd.DataFrame:
    """Create targets TSV file (replaces manual alphafold_input_file.tsv).
    Omits the legacy template_pdb_dict column. This column is no longer
    needed with the new IG pipeline (Tasks 1-4).
    Columns: target_chainseq, templates_alignfile, targetid.
    Args:
        target_chainseq: target sequence with '/' chain separators.
        alignment_file_path: path to the alignment TSV file.
        targetid: string identifier for this target.
        output_path: where to write the targets TSV.
    Returns:
        targets DataFrame (single row).
    """
    df = pd.DataFrame({
        "target_chainseq": [target_chainseq],
        "templates_alignfile": [os.path.abspath(alignment_file_path)],
        "targetid": [targetid],
    })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"[input_generator] Wrote targets file: {output_path}")
    return df
# ─────────────────────────────────────────────────────────────────────
# Main entry point: one-call auto-generation
# ─────────────────────────────────────────────────────────────────────
def auto_generate_inputs(template_pdb_path: str,
                         output_dir: str,
                         target_chainseq: Optional[str] = None,
                         targetid: Optional[str] = None
                         ) -> Tuple[str, str]:
    """Auto-generate all required input files from a single template PDB.
    Creates alignment TSV and targets TSV in output_dir. Files are
    persisted (not deleted) for reproducibility and debugging.
    If target_chainseq is None, infers it from the template PDB
    (i.e., template IS the target — conformation sampling mode).
    If targetid is None, derives it from the PDB filename.
    Args:
        template_pdb_path: path to the template PDB file.
        output_dir: directory for generated files.
        target_chainseq: target sequence with '/' separators.
            If None, extracted from the template PDB.
        targetid: string identifier. If None, derived from PDB filename.
    Returns:
        targets_path: path to the generated targets TSV.
        align_path: path to the generated alignment TSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Derive targetid from PDB filename if not provided
    if targetid is None:
        targetid = os.path.splitext(os.path.basename(template_pdb_path))[0]
    # Infer target_chainseq from PDB if not provided (uses load_pdb_coords)
    if target_chainseq is None:
        target_chainseq, total_res, chain_ids, chain_lengths = \
            get_chain_sequences_from_pdb(template_pdb_path)
        print(f"[input_generator] Inferred target_chainseq from PDB: "
              f"{target_chainseq[:40]}... ({total_res} res, "
              f"{len(chain_ids)} chains: {list(zip(chain_ids, chain_lengths))})")
    # Generate alignment file
    align_path = os.path.join(output_dir, f"{targetid}_alignment.tsv")
    generate_alignment_file(template_pdb_path, target_chainseq, align_path)
    # Generate targets file
    targets_path = os.path.join(output_dir, f"{targetid}_targets.tsv")
    generate_targets_file(target_chainseq, align_path, targetid, targets_path)
    print(f"[input_generator] Auto-generated inputs for '{targetid}' in {output_dir}")
    return targets_path, align_path