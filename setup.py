from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

version = '0.1'

setup(
    name='agios',
    packages=['agios'],
    version=version,
    install_requires=[requirement for requirement in requirements if len(requirement) > 0]
)
