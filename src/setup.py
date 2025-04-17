from setuptools import setup, find_packages

setup(
    name='meituan',
    version='0.1',
    packages=find_packages(include=['meituan', 'meituan.*']),
)