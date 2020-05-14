# Abstractive product summarization

If you are interested in our dataset, please fill out the [application form](https://drive.google.com/open?id=19gRK45LLczLxFg_n6qbW4gXyOOla9_wu) and email us.

## Requirement
Python2.7, PyTorch v0.4.1

## Training
To train the model with RAML:
```
cd raml
python main.py --mode train_raml
```

Then, to complete the training:
```
cd raml_aspect
python main.py --mode train --load-model ../raml/model.bin
```

## Test
```
python main.py --mode decode --output out.txt
```

## Acknowledgments
This repository is built upon [pcyin/pytorch_basic_nmt](https://github.com/pcyin/pytorch_basic_nmt)
