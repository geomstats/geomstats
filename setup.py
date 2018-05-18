from setuptools import setup, find_packages

with open('requirements.txt') as fp:
        install_requires = fp.read()

setup(name='geomstats',
      version='1.9',
      install_requires=install_requires,
      description='Geometric statistics on manifolds',
      url='http://github.com/ninamiolane/geomstats',
      author='Nina Miolane',
      author_email='ninamio78@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
