import numpy as np
import cv2
import sys
from random import randint
import csv
import imageio



fp = open('report.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

TEXT_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1.2

Source = './videos/Cars.mp4'


TITLE_TEXT_POSITION = (100, 40)

BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

def get_bgsubstructure(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()

    print('Invalid detector !')
    sys.exit(0)


cap = cv2.VideoCapture(Source)


bg_subtractor = []

original = []
t_gmg = []
t_mog = []
t_mog2 = []
t_knn = []
t_cnt = []

for i, a in enumerate(BGS_TYPES):
    #print(i, a)
    bg_subtractor.append(get_bgsubstructure(a))


def main():

    framecount = 0
    while True:
        ok, frame = cap.read()

        if not ok:
            print('End processing video')
            break

        framecount += 1

        frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        gmg = bg_subtractor[0].apply(frame)
        mog = bg_subtractor[1].apply(frame)
        mog2 = bg_subtractor[2].apply(frame)
        knn = bg_subtractor[3].apply(frame)
        cnt = bg_subtractor[4].apply(frame)

        gmg_count = np.count_nonzero(gmg)
        mog_count = np.count_nonzero(mog)
        mog2_count = np.count_nonzero(mog2)
        knn_count = np.count_nonzero(knn)
        cnt_count = np.count_nonzero(cnt)



        cv2.putText(mog, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(mog2, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(gmg, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(knn, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(cnt, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)


        cv2.imshow('Original', frame)

        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('GMG', gmg)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)

        original.append(frame)
        t_mog.append(mog)
        t_mog2.append(mog2)
        t_cnt.append(cnt)
        t_gmg.append(gmg)
        t_knn.append(knn)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

main()
imageio.mimsave('./gif/ori.gif', original, duration=20)
imageio.mimsave('./gif/mog.gif', t_mog, duration=20)
imageio.mimsave('./gif/mog2.gif', t_mog2, duration=20)
imageio.mimsave('./gif/knn.gif', t_knn, duration=20)
imageio.mimsave('./gif/cnt.gif', t_cnt, duration=20)
imageio.mimsave('./gif/gmg.gif', t_gmg, duration=20)
cap.release()
cv2.destroyAllWindows()