import os
from dotenv import load_dotenv

def load_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
    load_dotenv(dotenv_path, override=True)

    global KAFKA_URL, VIDEO_FPS, OUT_DIR, BASE_DIR

    KAFKA_URL = os.getenv("KAFKA_URL")
    VIDEO_FPS = int(os.getenv("VIDEO_FPS"))
    OUT_DIR = os.getenv("OUT_DIR")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_env()
