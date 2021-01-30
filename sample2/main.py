import cv2

def capture(vc):
    rval, frame = vc.read()
    if rval:
        frame = cv2.flip(frame, 1)
    return (rval, frame)


if __name__ == "__main__":
    vc = cv2.VideoCapture(1)
    if not vc.isOpened():
        exit - 1

    cv2.namedWindow("preview")

    rval, frame = capture(vc)
    while rval:
        rval, frame = capture(vc)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            print('Closing...')
            break
    ## finish
    vc.release()
    cv2.destroyWindow("preview")
