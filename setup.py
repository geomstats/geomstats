"""Create instructions to build the geomstats package."""
import os
import runpy
from itertools import chain

from setuptools import find_packages, setup


base_dir = os.path.dirname(os.path.abspath(__file__))
geomstats = runpy.run_path(os.path.join(base_dir, 'geomstats', '__init__.py'))


def parse_requirements_file(filename):
    """Read the lines of the requirements file."""
    with open(filename) as input_file:
        return input_file.read().splitlines()


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

    with open(os.path.join(base_dir, "README.md")) as f:
        long_description = f.read()

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
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Mathematics',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
        ],
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        data_files=[
            "LICENSE.md",
            "README.md",
        ],
        include_package_data=True,
        package_data={'': ['datasets/data/*',
                           'datasets/data/*/*',
                           'datasets/data/*/*/*',
                           'datasets/data/*/*/*/*']},
        zip_safe=False,
    )
