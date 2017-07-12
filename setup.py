from setuptools import setup


def read(file_path):
    with open(file_path) as f:
        return [line for line in f.readlines() if len(line) > 0]

version = 'dynamic-dev'

setup(
    name='agios',
    packages=['agios'],
    version=version,
    description='Simple genetic algorithms framework for Python',
    author='Michal Korman',
    author_email='m.korman94@gmail.com',
    url='https://github.com/mkorman9/agios',
    download_url='https://github.com/mkorman9/agios/tarball/{}'.format(version),
    keywords='ai genetic algorithms',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=read('requirements.txt'),
    tests_require=read('test-requirements.txt')
)
