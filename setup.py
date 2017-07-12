from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

version = 'dynamic-dev'

setup(
    name='agios',
    packages=['agios'],
    version=version,
    description='Simple genetic algorithms framework for Python',
    author='Michal Korman',
    author_email='m.korman94@gmail.com',
    url='https://github.com/mkorman9/agios',
    download_url='https://github.com/mkorman9/framepy/agios/{}'.format(version),
    keywords='ai genetic algorithms',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[requirement for requirement in requirements if len(requirement) > 0]
)
