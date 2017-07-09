from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

version = '1.0'

setup(
    name='genetics-algorithm',
    packages=find_packages(exclude=('tests',)),
    version=version,
    install_requires=[requirement for requirement in requirements if len(requirement) > 0]
)
