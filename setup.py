from setuptools import setup, find_packages
from geomstats.__about__ import __version__, install_requires, extras_require

setup(name='geomstats',
      version=__version__,
      install_requires=install_requires,
      extras_require=extras_require,
      description='Geometric statistics on manifolds',
      url='http://github.com/geomstats/geomstats',
      author='Nina Miolane',
      author_email='ninamio78@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
