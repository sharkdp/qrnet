# QRNet

A simple neural network which is able to predict the first digit that is stored
in a [QR-code](https://de.wikipedia.org/wiki/QR-Code).

## Generate training data

``` bash
mkdir train test
python create_qrcodes.py
```

## Training

Start the training via

``` bash
python train_qrnet.py
```

Then, run `tensorboard` to view the accuracy:

``` bash
tensorboard --logdir /tmp/qrnet-log --reload_interval 5
```
