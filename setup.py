from setuptools import setup

setup(
    name='BERT_GEC',
    version='0.0.1',    
    description='BERT GEC',
    author='Mark Newell',
    author_email='markcnewell700@gmail.com',
    packages=['gec'],
    install_requires=[
        "pytorch_pretrained_bert",
        "torch==1.11.0",
        "numpy",
        "cdifflib",
        "keras",
        "spacy==2.1.0"
    ],
    dependency_links=[
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz",
    ],
)