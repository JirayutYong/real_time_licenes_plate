import multiprocessing
import subprocess
import time

def run_license_plate_detector():
    subprocess.run(['python', 'main.jetson_brand.py'])

def run_folder_handler():
    subprocess.run(['python', 'upload_minio.py'])

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    # Running in separate processes
    pool.apply_async(run_license_plate_detector)
    pool.apply_async(run_folder_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()