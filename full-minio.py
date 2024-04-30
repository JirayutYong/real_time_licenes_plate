import io
import os
import time
from minio import Minio
from minio.error import S3Error
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import multiprocessing

minio_endpoint = '192.168.1.209:9080'
# minio_endpoint = '10.20.30.247:9080'
minio_access_key = 'Jettrack-66'
minio_secret_key = 'JettrackCEP-66'

class JSONFileHandler(FileSystemEventHandler):
    def __init__(self, minio_client, bucket_name, object_name):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.processed_files = set()

    def on_modified(self, event):
        if event.src_path.endswith('.json')and event.src_path not in self.processed_files:
            print(f"Detected modification in JSON file{event.src_path}")
            self.upload_to_minio(event.src_path)
            # time.sleep(1)

    def upload_to_minio(self, json_file_path):
        with open(json_file_path, 'rb') as file:
            json_data = file.read()

        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                print(f"Bucket created: {self.bucket_name}")
        except S3Error as e:
            print(f"Error checking/creating bucket: {e}")

        data_stream = io.BytesIO(json_data)
        try:
            self.minio_client.put_object(
                self.bucket_name,
                self.object_name,
                data_stream,
                len(json_data),
                content_type='application/json'
            )
            print(f"JSON file uploaded to Minio Successfully!! with #{self.object_name}#")
            #time.sleep(1)
        except S3Error as e:
            print("Error uploading JSON file to Minio: {e}")

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
            if os.path.isfile(event.src_path):  # Check if it's a file
                # print(f"New entry: {event.src_path}")
                new_file = event.src_path.replace('\\', '/')
                # print("new file >>", new_file)
                time.sleep(2)
                self.upload_minio(new_file)

    def upload_minio(self, new_file):
        # print("Its bucket > ", self.bucket_name)
        # print("its dest > ", self.destination_file)
        # print("its new file > ", new_file)
        pure_file_name = str(new_file.split('/')[-1])
        # print(pure_file_name)
        new_object = "{}/{}".format(self.destination_file,pure_file_name)
        self.client.fput_object(bucket_name=self.bucket_name,
                                object_name=new_object, file_path=new_file, content_type='image/jpeg')
        print(new_file, "successfully uploaded as object",
              new_object, "to bucket", self.bucket_name,
              )

def run_json_file_watcher(json_file_path):
    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False  # Set to True if using HTTPS
    )

    json_handler = JSONFileHandler(
        minio_client, 'json-files', 'details')
    json_observer = Observer()
    json_observer.schedule(json_handler,
                           path=json_file_path.rsplit("/", 1)[0], recursive=False)
    json_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        json_observer.stop()
    json_observer.join()

def run_folder_watcher(folder_path):
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )

    bucket_name = "images"
    if folder_path == "./save_car":
        destination_folder = "/cars"
    elif folder_path == "./save_license_plate":
        destination_folder = "/licenses"
    else:
        return "folder_path missing"

    # print("its folder path >", folder_path)
    # print("its client >", client)
    # print("Its bucket >", bucket_name)
    # print("its dest >", destination_folder)

    
    folder_handler = FolderHandler(client, bucket_name, destination_folder)
    folder_observer = Observer()
    folder_observer.schedule(folder_handler, folder_path, recursive=False)
    folder_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        folder_observer.stop()
    folder_observer.join()

if __name__ == "__main__":
    print("Waiting Events")
    # json file handling process
    json_process = multiprocessing.Process(target=run_json_file_watcher, args=('./save_json/data.json',))
    json_process.start()

    # folder handling process
    car_folder_process = multiprocessing.Process(target=run_folder_watcher, args=('./save_car',))
    license_folder_process = multiprocessing.Process(target=run_folder_watcher, args=('./save_license_plate',))

    car_folder_process.start()
    license_folder_process.start()

    try:
        json_process.join()
        car_folder_process.join()
        license_folder_process.join()
    except KeyboardInterrupt:
        json_process.terminate()
        car_folder_process.terminate()
        license_folder_process.terminate()


# images/cars/name_img
# images/licenses/name_img
