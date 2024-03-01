from threading import Thread
import cv2

from cv2.typing import MatLike
from typing import Tuple


class VideoCapture:
    def __init__(self, sizes: Tuple[int, int] = None, source: any = 0) -> None:
        self.video = cv2.VideoCapture(source)
        (self.grabbed, self.frame) = self.video.read()

        self.sizes = sizes
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self) -> None:
        while True:
            if self.stopped:
                break
            (self.grabbed, frame) = self.video.read()
            if not self.grabbed:
                self.stop()
                break

            if self.sizes:
                self.frame = cv2.resize(frame, self.sizes)

    def read(self) -> MatLike:
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        self.video.release()

    def is_stop(self):
        return self.stopped
