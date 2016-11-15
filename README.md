# QRNet

A simple neural network that is able to predict the
first character (`[A-Za-z0-9]`) that is stored in a
[QR-code](https://de.wikipedia.org/wiki/QR-Code).

## Generate training data

Generate QR-codes from random strings for training and validation:
``` bash
python create_qrcodes.py
```
It creates QR-codes that encode *10* random characters, such as this:

![Sample QR-code](00bLF6qvQF.png)

## Training

Start the training:
``` bash
python train_qrnet.py
```

Then, run `tensorboard` to view the test-set accuracy:
``` bash
tensorboard --logdir /tmp/qrnet-log --reload_interval 5
```

After approximately *400.000* epochs (with *20* images per batch), it reaches a
test-set accuracy of over *0.999*.
