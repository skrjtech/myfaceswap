import cv2
import argparse

def Play(path):
    video = cv2.VideoCapture(path)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path-dir', type=str, default='./output.mp4', help='(default: ./output.mp4)')
    parse = p.parse_args()
    Play(parse.path_dir)
    