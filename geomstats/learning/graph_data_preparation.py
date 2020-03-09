"""Prepare and Process Graph-structured data."""


class Graph():
    """Class for generating a graph object from a dataset.

    Prepare Graph object from a dataset file.

    Parameters
    ----------
    Graph_Matrix_Path : Path to graph adjacency matrix.

    Labels_Path : Path to labels of the nodes of the graph.
    """

    edges = None
    labels = None

    def __init__(self,
                 Graph_Matrix_Path=r'examples\data_example'
                                   r'\graph_random\Graph_Example_Random.txt',
                 Labels_Path=r'examples\data_example'
                             r'\graph_random\Graph_Example_Random_Labels.txt'):
        self.edges = {}
        with open(Graph_Matrix_Path, "r") as edges_file:
            for i, line in enumerate(edges_file):
                lsp = line.split()
                self.edges[i] = [k for k, value in
                                 enumerate(lsp) if (int(value) == 1)]

        if Labels_Path is not None:
            self.labels = {}
            with open(Labels_Path, "r") as label_file:
                for i, line in enumerate(label_file):
                    self.labels[i] = []
                    self.labels[i].append(int(line))
