"""Prepare and process graph-structured data."""

from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor


def build_window_trees(file_path, n_windows=106):
    """Load data from fasta files.

    Parameters
    ----------
    n_windows : int
        Number of windows to split the data into. 106 splits it about
        into how many genes there are.
    file_path : str
        Filepath of the aligned, concatenated genes.

    Returns
    -------
    trees : list[Bio.Phylo.BaseTree.Tree]
        Trees representing the phylogenetic relationships inferred from
        the Neighbor-Joining algorithm.

    References
    ----------
    .. [Rokas2003]  Rokas, A., Williams, B., King, N. et al. Genome-scale
      approaches to resolving incongruence in molecular phylogenies.
      Nature 425, 798–804 (2003). https://doi.org/10.1038/nature02053
    """
    full_aln = AlignIO.read(file_path, "fasta")
    total_bp = full_aln.get_alignment_length()

    window_bp = total_bp // n_windows

    calculator = DistanceCalculator("identity")
    constructor = DistanceTreeConstructor(calculator, "nj")

    window_trees = []

    for i in range(n_windows):
        start = i * window_bp
        end = start + window_bp if i < n_windows - 1 else total_bp
        w_aln = full_aln[:, start:end]
        bp_tree = constructor.nj(calculator.get_distance(w_aln))
        window_trees.append(bp_tree)

    return window_trees
