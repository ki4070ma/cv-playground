import cv2
import numpy as np


def makeResult(frame, flow):
    step = 16
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = grayFrame.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # vis = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
    vis = frame

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def check_motion(pre, curr):
    prevgray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    val = np.sum(np.abs(flow)) / np.prod(gray.shape)
    print(val)
    return flow, val


if __name__ == "__main__":

    # vid_src = 1  # camera device
    vid_src = "./video.mp4"  # video file
    vid_src = "./video_04:00-04:03.mp4"  # video file

    vc = cv2.VideoCapture(vid_src)
    if not vc.isOpened():
        exit - 1

    cv2.namedWindow("preview")

    rval, frame = vc.read()
    prev_frame = frame
    h, w = prev_frame.shape[:2]
    size = (int(w / 2), int(h / 2))
    prev_frame = cv2.resize(prev_frame, size)

    # For video write
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # write_size = (width, height)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(vc.get(cv2.CAP_PROP_FPS))
    # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # High quality, large size
    fmt = cv2.VideoWriter_fourcc(*"H264")  # Low quality, small size
    writer = cv2.VideoWriter("./out.mp4", fmt, frame_rate, size)

    of_list = []

    for i in range(frame_count):
        print(i)
        if i == frame_count - 1:
            break
        rval, frame = vc.read()
        frame = cv2.resize(frame, size)
        flow, val = check_motion(prev_frame, frame)
        of_list.append(val)
        # cv2.imshow("preview", frame)
        # cv2.imshow("preview", makeResult(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow))

        out_frame = makeResult(frame, flow)
        cv2.imshow("preview", out_frame)
        writer.write(out_frame)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            print("Closing...")
            break
        prev_frame = frame

    # Normalization
    max_val = max(of_list)
    print("*" * 10)
    with open("./of_list.txt", "w") as fw:
        for v in ["{:.2f}".format(x / max_val) for x in of_list]:
            fw.write(v + "\n")

    ## finish
    writer.release()
    vc.release()
    cv2.destroyWindow("preview")
