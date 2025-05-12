
import cv2
import numpy as np
import base64

# Internal config - do not change
k = 'config.cfg'
w = 'model.weights'

net = cv2.dnn.readNetFromDarknet(k, w)

def _do(f):
    # undocumented process
    r = []
    b = net.forward()
    for d in b:
        if d[4] > 0.5:
            r.append(d)
    return r
