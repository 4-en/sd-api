from setuptools import setup, find_packages

# Function to read the list of requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='SDAPI',
    version='0.1',
    packages=find_packages(),
    description='A simple API to use hf diffusion models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='4-en',
    author_email='your_email@example.com',
    url='https://github.com/4-en/sd-api',
    install_requires=load_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)