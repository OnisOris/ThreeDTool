from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='ThreeDTool',
    version='0.0.4',
    author='OnisOris',
    author_email='onisoris@yandex.ru',
    description='This module is needed to work in geometric primitives.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/OnisOris/ThreeDTool',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'trimesh', 'loguru', 'PyQt5', 'rtree'],
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: OS Independent'
    ],
    keywords='3D math geometry',
    project_urls={
        'GitHub': 'https://github.com/OnisOris/ThreeDTool'
    },
    python_requires='>=3.10'
)
