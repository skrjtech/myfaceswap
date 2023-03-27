from enum import Enum

def video_filepath(device_id, producer_time):
    return f"temp/{device_id}/{producer_time}.mkv"

class VideoWriterState(Enum):
    OPEN = 0
    RELEASED = 1

import cv2
import time

def estimate_capture_time(capture_time_post, time_offset):
    capture_time_estimate = capture_time_post - time_offset
    return capture_time_estimate

class Camera():
    def __init__(self, idx_cam, width, height, fps, time_offset):
        self._idx_cam = idx_cam
        self._width = width
        self._height = height
        self._fps = fps
        self._time_offset = time_offset

        capture = cv2.VideoCapture(idx_cam)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        print(capture.get(cv2.CAP_PROP_FRAME_WIDTH), width)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == width
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), height)
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == height
        capture.set(cv2.CAP_PROP_FPS, fps)
        print(capture.get(cv2.CAP_PROP_FPS), fps)
        assert capture.get(cv2.CAP_PROP_FPS) == fps
        self._capture = capture

    def read(self):
        # capture_time_pre = time.time()
        ret, frame = self._capture.read()
        capture_time_post = time.time()

        capture_time_estimate = estimate_capture_time(capture_time_post, self._time_offset)

        return ret, frame, capture_time_estimate
    
import os
import cv2

def create_command(codec, width, height, fps, pixel_format, filename, bitrate=None):
    # pixel_format : ex) "bgr24"
    # codec        : ex) "h264"
    # bitrate      : ex) "800k"

    dimension = f'{width}x{height}'
    # filename_os = f'"{os.path.abspath(filename)}"'
    if not codec in ["H264", "h264"]:
        raise NotImplementedError("Not Supported H264 in gstreamer_video_writer.py")

    command = [
        f"appsrc",
        f"autovideoconvert",
        f"omxh264enc" if bitrate is None else f"omxh264enc bitrate={int(bitrate[:-1])*1024}",
        f"matroskamux",
        f"filesink location={filename} sync=false",
        "video/x-raw,format=I420",
        "v4l2sink device=/dev/video42",
    ]
    command_all = " ! ".join(command)

    return command_all

class GstreamerVideoWriter():
    def __init__(self, device_id, producer_time, codec_forcc, width, height, fps, pixel_format, bitrate=None):
        self._producer_time = producer_time
        self._codec_forcc = codec_forcc
        self._width = width
        self._height = height
        self._fps = fps

        filename = video_filepath(device_id, producer_time)
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._filename = filename

        command = create_command(codec_forcc, width, height, fps, pixel_format, filename, bitrate)
        self._command = command
        self._writer = cv2.VideoWriter(command, 10, fps, (width, height))
        print("".join(command))

        self._frame_count = 0
        self._state = VideoWriterState.OPEN

    def write(self, frame):
        if self._state == VideoWriterState.OPEN:
            self._writer.write(frame)
            # self._proc.stdin.write(frame.tostring())
            self._frame_count += 1

    def release(self):
        if not self._state == VideoWriterState.RELEASED:
            # self._writer.release()
            self._writer.release()
            self._state = VideoWriterState.RELEASED
        return self._filename, self._frame_count

if __name__ == "__main__":
    CAMERA_DEVICE_INDEX = 0
    CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
    CAPTURE_FPS = 30
    CAPTURE_TIME_OFFSET = 0.17
    CAPTURE_PIXEL_FORMAT = "bgr24"
    WRITER_CODEC_FOURCC = "H264"
    WRITER_BITRATE = "800k"

    import sys
    import time

    CAMERA_DEVICE_INDEX = int(sys.argv[1]) if len(sys.argv) >= 2 else CAMERA_DEVICE_INDEX

    writer = GstreamerVideoWriter(CAMERA_DEVICE_INDEX, int(time.time()), WRITER_CODEC_FOURCC, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS, CAPTURE_PIXEL_FORMAT, WRITER_BITRATE)
    camera = Camera(CAMERA_DEVICE_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS, CAPTURE_TIME_OFFSET)

    n_frame_to_capture = 30 * 1 * 60 * 2
    i_frame = 0
    while True:
        ret, frame, capture_time_estimate = camera.read()
        writer.write(frame)

        i_frame += 1
        if i_frame >= n_frame_to_capture:
            break

    filename, frame_count = writer.release()