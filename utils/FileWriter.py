import os
from datetime import datetime
import cv2

from config import config


class FileWriter:

    def __init__(this, directory) -> None:
        this.full_path = os.path.join(config.BASE_DIR, config.OUT_DIR, directory)

        if not os.path.exists(this.full_path):
            os.makedirs(this.full_path)

    def write_jpg(this, image, prefix):
        file_name = this.get_file_name(prefix, ".jpeg")
        cv2.imwrite(file_name, image)
        return file_name
    # cv2.imwrite(os.path.join("./capture_car_to_pred2", "car_daw.jpeg"), image)

    def get_file_name(this, prefix, format):
        stamp = this.get_file_date_time()
        path_name = os.path.join(this.full_path, f"{prefix}{stamp}{format}")
        return path_name

    def get_file_date_time(this):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
