import cv2 as cv
import numpy as np
from os import listdir, mkdir
from os.path import join, dirname, realpath
from time import time
import functions2 as f
import matplotlib.pyplot as plt
import pandas as pd
import data


def main(v):
    if not v in listdir(dirname(realpath(__file__))) or input('run (y/n)') == 'y':
        try:
            mkdir(join(dirname(realpath(__file__)), v))
        except:
            pass
        cap = cv.VideoCapture(join(dirname(realpath(__file__)), f'{v}.MOV'))
        t1 = time()
        lframe = np.zeros((1920, 1080), dtype='uint8')
        i = 0
        while i <= 650:
            cap.read()
            i += 1
        ret, frame = cap.read()
        frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        lframe = d = cv.threshold(frameg, 100, 255, cv.THRESH_BINARY_INV)[1]

        i = []
        j = 0
        t = []
        y = []
        while j <= 2300:
            ret, frame = cap.read()
            t2 = time()
            blank = np.zeros((1920, 1080, 3), dtype='uint8')
            kernel_1 = np.ones((3, 3), np.uint8)
            frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.putText(blank, f'{round(1 / (t2 - t1), 2)}fps', (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv.LINE_AA)
            d = cv.threshold(frameg, 100, 255, cv.THRESH_BINARY_INV)[1]

            dif = cv.absdiff(d, lframe)

            dif = cv.threshold(dif, 100, 255, cv.THRESH_BINARY_INV)[1]
            lframe = d
            '''
            dif = cv.erode( dif, kernel_1, iterations = 2)
            dif = cv.dilate(dif, kernel_1, iterations = 2)
            dif = cv.GaussianBlur(dif,(3,3),cv.BORDER_DEFAULT)
            '''
            dif = cv.bitwise_not(dif)
            dif = cv.erode(dif, kernel_1, iterations=1)
            dif = cv.dilate(dif, kernel_1, iterations=2)
            dif = cv.GaussianBlur(dif, (15, 15), cv.BORDER_DEFAULT)
            cont = cv.findContours(dif, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
            try:
                cm = f.contmax(cont)
                i.append(cv.contourArea(cm))
                if cv.contourArea(cm) >= sum(i) / len(i) - sum(i) / (8 * len(i)):
                    cv.drawContours(blank, cm, -1, (0, 255, 0), 2)

            except:
                pass
            # shows
            if cv.contourArea(cm) >= sum(i) / len(i) - sum(i) / (8 * len(i)):
                y.append((1920 - cv.boundingRect(cm)[1]) * (49 / 120))
                t.append(j / 120)

            t1 = t2
            j += 1
            f.imshows(frameg, d, dif, blank)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        if input('save(y/n)') == 'y':
            L = [[t[i], y[i]] for i in range(len(t))]
            df = pd.DataFrame(L, index=None, columns=['t', 'y'])
            df.to_csv(join(dirname(realpath(__file__)), join(v, f'{v}.csv')))
    data.main(v)


if __name__ == '__main__':

    if int(input('')) == 1:
        main('m1')
    else:
        main('m2')
