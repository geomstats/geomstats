"""Prepare and Process Data."""
import io


class Random_walk_graph():
    """Generate random walks on a graph."""
    def __init__(self, X, Y, path=True):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y
        self.k = 0
        self.path = path
        self.p_c = 1

    def set_walk(self, maximum_walk, continue_probability):
        self.k = maximum_walk
        self.p_c = continue_probability

    def set_path(self, path_val):
        self.path = path_val

    def light_copy(self):
        rwc_copy = RandomWalkCorpus(self.X, self.Y, path=self.path)
        rwc_copy.k = self.k
        rwc_copy.p_c = self.p_c

        return rwc_copy

    def getFrequency(self):
        return torch.Tensor([[k, len(v)] for k, v in self.X.items()])

    def _walk(self, index):
        path = []
        c_index = index
        path.append(c_index)
        for i in range(self.k):

            if (random.random() > self.p_c):
                break
            c_index = self.X[c_index][random.randint(0, len(self.X[c_index]) - 1)]
            path.append(c_index)
        return path if (self.path) else [c_index]

    def __getitem__(self, index):
        return torch.LongTensor([self._walk(index)]), torch.LongTensor(self.Y[index])

    def __len__(self):
        return len(self.X)

class Graph():



    def load_from_file(Graph_Matrix_Path='examples/data_example/Graph_Example_Random.txt',
                       Labels_Path= 'examples/data_example/Graph_example_Random_Labels.txt'):
        """Loads a Graph from a file."""

        edges = {}
        with io.open(Graph_Matrix_Path, "r") as edges_file:
            for i, line in enumerate(edges_file):
                lsp = line.split()
                edges[i] = [k for k, value in enumerate(lsp) if (int(value) == 1)]

        print('edges', edges)

        if Labels_Path != None:
            labels = {}
            with io.open(Labels_Path, "r") as label_file:
                for i, line in enumerate(label_file):
                    labels[i] = []
                    labels[i].append(int(line))

            print('labels', labels)





