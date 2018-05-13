# [Coarse-to-Fine Decoding for Neural Semantic Parsing](http://homepages.inf.ed.ac.uk/s1478528/acl18-coarse2fine.pdf)

## Setup

### Requirements

- Python 3.5
- [PyTorch 0.2.0.post3](https://pytorch.org/previous-versions/) (GPU)

### Install Python dependency

```sh
pip install -r requirements.txt
```

### Download data and pretrained models

Download the zip file from [Google Drive](https://drive.google.com/file/d/18oMNo4yC01gwMjHcfmE-_G5qE7X5SLYt/view?usp=sharing), and copy it to the root folder.

```sh
unzip acl18coarse2fine_data_model.zip
```

## Usage

### Run pretrained models

```sh
./pretrain.sh [geoqueries|atis|django|wikisql] GPU_ID
```

### Run experiments

```sh
./run.sh [geoqueries|atis|django|wikisql] GPU_ID
```

## Acknowledgments

- The implementation is based on [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- The preprocessing and evaluation code used for WikiSQL is from [salesforce/WikiSQL](https://github.com/salesforce/WikiSQL).
