from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='agios-examples',
    version='1.0',
    install_requires=[requirement for requirement in requirements if len(requirement) > 0]
)
