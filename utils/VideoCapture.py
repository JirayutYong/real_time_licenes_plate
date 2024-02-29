from threading import Thread
import cv2

from cv2.typing import MatLike
from typing import Tuple


class VideoCapture:
    def __init__(self, sizes: Tuple[int, int], source: any = 0) -> None:
        self.video = cv2.VideoCapture(source)
        (self.grabbed, self.frame) = self.video.read()

        self.sizes = sizes
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self) -> None:
        while True:
            if self.stopped:
                return
            (self.grabbed, frame) = self.video.read()

            if not self.grabbed:
                print("stopped")
                self.stop()
                return

            self.frame = cv2.resize(frame, self.sizes)

    def read(self) -> MatLike:
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        self.video.release()
