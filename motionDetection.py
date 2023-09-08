import numpy as np
import cv2
import sys
from random import randint
import imageio



TEXT_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))
TRACKER_COLOR =  (randint(0,255), randint(0, 255), randint(0, 255))


FONT = cv2.FONT_HERSHEY_SIMPLEX
Source = './videos/Animal_1.mp4'



BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']





def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel




get_kernel('closing')



def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, get_kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
        dilation =  cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation


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

motion_image_lst = []
noBG_image_lst = []
bg_subtractor = get_bgsubstructure(BGS_TYPES[1])



BGS_TYPE = BGS_TYPES[1]

minArea = 250

def main():
    while True:
        ok, frame = cap.read()

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        bg_mask = bg_subtractor.apply(frame)
        #bg_mask = get_filter(bg_mask, 'combine')
        bg_mask  = cv2.medianBlur(bg_mask, 3)

        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                x, y,  w , h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (10, 30), (250, 55), (255,0,0), -1)
                cv2.putText(frame, 'Motion detected!', (10,50),FONT,0.8,TEXT_COLOR,2)
                cv2.drawContours(frame, cnt, -1, TRACKER_COLOR, 3)
                cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,255), 10)

        results = cv2.bitwise_and(frame, frame, mask=bg_mask)
        if not ok:
            print('End processing video')
            break


        cv2.imshow('Frame', frame)
        cv2.imshow('Masked Frame', results)  # Show the masked frame

        motion_image_lst.append(frame)
        noBG_image_lst.append(results)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()
imageio.mimsave('./gif/motion.gif', motion_image_lst[70:], duration=20)
imageio.mimsave('./gif/noBG.gif', noBG_image_lst[70:], duration=20)
cap.release()
cv2.destroyAllWindows()

