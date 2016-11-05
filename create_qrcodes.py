import qrcode
import os.path
from random import shuffle


def saveQRcode(folder, num):
    # The 'data' will be the number, rendered as a string with leading zeros
    data = "{:04}".format(num)
    print(data)

    qr = qrcode.QRCode(version=1, box_size=1, border=0)
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image()
    img.save(os.path.join(folder, data + ".png"))

allNums = list(range(0, 10000))
shuffle(allNums)

trainNums = allNums[0:8000]
testNums = allNums[8000:]

for n in trainNums:
    saveQRcode("train", n)

for n in testNums:
    saveQRcode("test", n)
