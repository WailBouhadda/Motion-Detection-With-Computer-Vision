import numpy as np
import cv2
import sys
from random import randint
import imageio



TEXT_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0,255), randint(0, 255), randint(0, 255))
print(TEXT_COLOR)
print(BORDER_COLOR)


FONT = cv2.FONT_HERSHEY_SIMPLEX
Source = './videos/Pedestrians_2.mp4'



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
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True, varThreshold=100)
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                        maxPixelStability=15 * 60, isParallel=True)

    print('Invalid detector !')
    sys.exit(0)





cap = cv2.VideoCapture(Source)



bg_subtractor = get_bgsubstructure(BGS_TYPES[4])



BGS_TYPE = BGS_TYPES[0]


original = []
bgmask = []
opening = []
closing = []
dilation = []
combine = []

def main():
    while True:
        ok, frame = cap.read()

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        bg_mask = bg_subtractor.apply(frame)
        fg_mask_dilation = get_filter(bg_mask, 'dilation')
        fg_mask_closing = get_filter(bg_mask, 'closing')
        fg_mask_opening = get_filter(bg_mask, 'opening')
        fg_mask_combine = get_filter(bg_mask, 'combine')

        print(frame.shape)

        if not ok:
            print('End processing video')
            break

        res_dilation = cv2.bitwise_and(frame, frame, mask=fg_mask_dilation)
        res_closing = cv2.bitwise_and(frame, frame, mask=fg_mask_closing)
        res_opening = cv2.bitwise_and(frame, frame, mask=fg_mask_opening)
        res_combine = cv2.bitwise_and(frame, frame, mask=fg_mask_combine)

        #cv2.putText(res_combine, 'Background substraction with ' + BGS_TYPE, (10,50), FONT, 1, BORDER_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, 'Original with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        cv2.putText(bg_mask, 'Mask with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        cv2.putText(res_combine, 'Combined with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        cv2.putText(res_dilation, 'Dilation with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        cv2.putText(res_opening, 'Opening with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        cv2.putText(res_closing, 'Closing with ' + BGS_TYPE, (10, 50), FONT, 1, BORDER_COLOR, 2,
                    cv2.LINE_AA)
        #if BGS_TYPE != 'MOG' and BGS_TYPES != 'GMG':
           # cv2.imshow('Background', bg_subtractor.getBackgroundImage())

        cv2.imshow('Frame', frame)
        cv2.imshow('Masked Frame', bg_mask)  # Show the masked frame
        cv2.imshow('dilation', res_dilation)  # Show the masked frame
        cv2.imshow('opening', res_opening)  # Show the masked frame
        cv2.imshow('closing', res_closing)  # Show the masked frame
        cv2.imshow('combine', res_combine)  # Show the masked frame

        original.append(frame)
        bgmask.append(bg_mask)
        opening.append(res_opening)
        closing.append(res_closing)
        dilation.append(res_dilation)
        combine.append(res_combine)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()
imageio.mimsave('./gif/original.gif', original, duration=20)
imageio.mimsave('./gif/bgmask.gif', bgmask, duration=20)
imageio.mimsave('./gif/closing.gif', closing, duration=20)
imageio.mimsave('./gif/opening.gif', opening, duration=20)
imageio.mimsave('./gif/combine.gif', combine, duration=20)
imageio.mimsave('./gif/dilation.gif', dilation, duration=20)

cap.release()
cv2.destroyAllWindows()