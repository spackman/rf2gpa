from setuptools import setup, find_packages

setup(
    name='mines-dac-pressure',
    version='1.0.0',
    description='CLI tool for DAC ruby fluorescence pressure analysis',
    author='Isaac Spackman, Kacy Mendoza',
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "matplotlib", "scipy", "lmfit"
    ],
    entry_points={
        'console_scripts': [
            'dac-pressure=mines_dac.cli:main',
        ],
    },
)
