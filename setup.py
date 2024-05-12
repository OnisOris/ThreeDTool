from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(
  name='ThreeDTool',
  version='0.0.1',
  author='OnisOris',
  author_email='onisoris@yandex.ru',
  description='This module is needed to work in geometric primitives.',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/OnisOris/ThreeDTool',
  packages=find_packages(),
  install_requires=['numpy', 'matplotlib'],
  classifiers=[
    'Programming Language :: Python :: 3.12',
    'License :: OSI Approved :: GPL-2.0',
    'Operating System :: OS Independent'
  ],
  keywords='files speedfiles ',
  project_urls={
    'GitHub': 'your_github'
  },
  python_requires='>=3.6'
)