# Abstractive product summarization

Pytorch implementation for our AAAI-2020 paper: Aspect-Aware Multimodal Summarization for Chinese E-Commerce Products

If you are interested in our dataset, please fill out the [application form](https://github.com/hrlinlp/cepsum/files/8858425/application.form.pdf) and email us.



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
