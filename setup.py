import os
import runpy
from itertools import chain

from setuptools import find_packages, setup


base_dir = os.path.dirname(os.path.abspath(__file__))
geomstats = runpy.run_path(os.path.join(base_dir, 'geomstats', '__init__.py'))


def parse_requirements_file(filename):
    with open(filename) as f:
        return f.read().splitlines()


if __name__ == '__main__':
    requirements = parse_requirements_file('requirements.txt')

    install_requires = []
    optional_dependencies = {}
    for requirement in requirements:
        # TensorFlow and PyTorch are optional dependencies.
        if 'torch' in requirement or 'tensorflow' in requirement:
            package = requirement.split('>')[0].split('=')[0]
            optional_dependencies[package] = [requirement]
        else:
            install_requires.append(requirement)

    dev_requirements = parse_requirements_file('dev-requirements.txt')
    extras_require = {
        'test': dev_requirements,
        **optional_dependencies
    }
    extras_require['all'] = list(chain(*extras_require.values()))

    setup(
        name='geomstats',
        version=geomstats['__version__'],
        install_requires=install_requires,
        extras_require=extras_require,
        description='Geometric statistics on manifolds',
        url='http://github.com/geomstats/geomstats',
        author='Nina Miolane',
        author_email='ninamio78@gmail.com',
        license='MIT',
        packages=find_packages(),
        zip_safe=False
    )
