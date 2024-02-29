import time
from enum import Enum
from typing import Callable, Tuple, List


class FPS_KEY(Enum):
    CONTINUE = "CONTINUE"
    STOP = "STOP"


class FPS:
    def __init__(self, fps):
        self._fps = fps
        self.frame_count = 0
        self.subscriber: List[Tuple[int, any]] = []
        pass

    def run_with_non_block_fps(self, runner: Callable[[], FPS_KEY]):
        delta = 0
        stamp = time.time()

        time_per_frame = 1 / self._fps

        while True:
            if delta > time_per_frame:
                if runner() == FPS_KEY.STOP:
                    break
                self.frame_count += 1
                print(self.frame_count)

                delta = 0

            newStamp = time.time()
            delta += newStamp - stamp
            stamp = newStamp

    def run_with_block_fps(self, runner: Callable[[], FPS_KEY]):
        delay_per_frame = 1.0 / self._fps

        while True:
            start_time = time.time()
            elapsed_time = time.time() - start_time

            if runner() == FPS_KEY.STOP:
                break

            self.__run_subscribed_event()
            self.frame_count += 1
            print(self.frame_count)

            remaining_time = delay_per_frame - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)


    def subscribe(self, runner=None, every=1):
        self.subscriber.append((every, runner))

    # private
    def __run_subscribed_event(self):
        for every, event in self.subscriber:
            if self.frame_count % every == 0:
                event()