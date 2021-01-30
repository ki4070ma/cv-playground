import cv2
import numpy as np


def makeResult(grayFrame, flow):
    step = 16
    h, w = grayFrame.shape[:2]
    y, x = np.mgrid[
           step // 2: h: step, step // 2: w: step
           ].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def capture(vc):
    rval, frame = vc.read()
    if rval:
        frame = cv2.flip(frame, 1)
    return (rval, frame)


def check_motion(pre, curr):
    prevgray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(np.sum(np.abs(flow)) / np.prod(gray.shape))
    return flow, np.sum(np.abs(flow)) / np.prod(gray.shape) > 1.0





if __name__ == "__main__":
    vc = cv2.VideoCapture(1)
    if not vc.isOpened():
        exit - 1

    cv2.namedWindow("preview")

    rval, frame = capture(vc)
    prev_frame = frame
    h, w = prev_frame.shape[:2]
    print(h, w)
    size = (int(w / 2), int(h / 2))
    print(size)
    prev_frame = cv2.resize(prev_frame, size)
    while rval:
        rval, frame = capture(vc)
        frame = cv2.resize(frame, size)
        flow, _ = check_motion(prev_frame, frame)
        # cv2.imshow("preview", frame)
        cv2.imshow("preview", makeResult(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow))
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            print("Closing...")
            break
        prev_frame = frame

    ## finish
    vc.release()
    cv2.destroyWindow("preview")
