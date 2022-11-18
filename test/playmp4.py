import cv2

videoCap = cv2.VideoCapture('../base1.mp4')
WIDTH = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
MAXFRAMENUM = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

if (videoCap.isOpened() == False):
    print("ビデオファイルを開くとエラーが発生しました")

while (videoCap.isOpened()):

    ret, frame = videoCap.read()
    if ret == True:

        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

videoCap.release()
cv2.destroyAllWindows()