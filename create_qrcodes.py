import qrcode
import os.path
import random
import string
import multiprocessing


def randomString():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def saveQRcode(folder):
    data = randomString()
    print("Create QR-code with data='{}'".format(data))

    qr = qrcode.QRCode(version=1, box_size=1, border=0)
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image()
    img.save(os.path.join(folder, data + ".png"))


N_TRAIN = 50000
N_TEST = 5000

pool = multiprocessing.Pool(6)
pool.map(saveQRcode, ("train" for _ in range(N_TRAIN)))
pool.map(saveQRcode, ("test" for _ in range(N_TEST)))
