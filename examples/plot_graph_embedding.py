"""
Embed a graph into a Manifold and plot
the result.
"""
import os

from geomstats.learning.data_preparation import *

def main():

    G = Graph
    G.load_from_file()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] != 'numpy':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()