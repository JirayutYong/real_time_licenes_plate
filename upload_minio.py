# -*- coding: utf-8 -*-
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from minio import Minio
import multiprocessing


class FolderHandler(FileSystemEventHandler):
    def __init__(self, client, bucket_name, destination_file):
        self.client = client
        self.bucket_name = bucket_name
        self.destination_file = destination_file

    def on_any_event(self, event):
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)
            print("Created bucket", self.bucket_name)
        else:
            print("Bucket", self.bucket_name, "already exists")
        if event.is_directory or event.event_type == 'created':
            # print("New entry: {}".format(event.src_path))
            new_file = event.src_path.replace('\\', '/')
            print("new file >>", new_file)
            self.upload_minio(new_file)

    def upload_minio(self, new_file):
        # print("Its bucket > ", self.bucket_name)
        # print("its dest > ", self.destination_file)
        # print("its new file > ", new_file)

        if os.path.isfile(new_file):
            pure_file_name = str(new_file.split('/')[-1])
            print(pure_file_name)
            new_object = "{}/{}".format(self.destination_file, pure_file_name)
            self.client.fput_object(bucket_name=self.bucket_name,
                                    object_name=new_object, file_path=new_file)
            print(new_file, "successfully uploaded as object",
                  new_object, "to bucket", self.bucket_name)
        else:
            print("Skipping directory: {}".format(new_file))


def folder_checker(folder_path):
    # print("its folder path >", folder_path)
    client = Minio("10.20.30.247:9000",
                   access_key="Jettrack-66",
                   secret_key="JettrackCEP-66",
                   secure=False
                   )
    bucket_name = "images"

    if folder_path == "./save_car":
        destination_folder = "/cars"
    elif folder_path == "./save_license_plate":
        destination_folder = "/licenses"
    else:
        return "folder_path missing"

    # print("its client >", client)
    # print("Its bucket >", bucket_name)
    # print("its dest >", destination_folder)
    print("waiting event")
    event_handler = FolderHandler(client, bucket_name, destination_folder)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_folder_checker(folder_path):
    folder_checker(folder_path)


if __name__ == "__main__":
    car_folder_path = "./save_car"
    license_folder_path = "./save_license_plate"

    pool = multiprocessing.Pool()

    pool.map(run_folder_checker, [car_folder_path, license_folder_path])

    # Close the pool to free up resources
    pool.close()
    pool.join()
