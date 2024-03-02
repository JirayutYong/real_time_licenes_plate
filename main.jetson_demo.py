import pandas as pd
from tracker import *
from datetime import datetime
from ultralytics import YOLO
import cv2
import shutil
import torch
from sort.sort import *
from util import get_car
import json

from utils.videoCapture import VideoCapture
from utils.fps import (FPS, FPS_KEY)
from utils.fileWriter import FileWriter

from config import config


# Set Path file
input_test = "./capture_car_to_pred"
input_car = "./save_car"
input_license_plate = "./save_license_plate"
json_file_path = "./save_json/data.json"

now = datetime.now()
stamp_day = None
stamp_time = None


car_model = YOLO("./models/yolov8n.pt")
license_plate_detector = YOLO("./models/license_plate_detector.pt")
license_plate_recognition = YOLO("./models/province.pt")
car_brand_detector = YOLO("./models/brand.pt")

# Check for CUDA device and set i./save_license_platet
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using device: {}'.format(device))


# # Use CUDA
# car_model.to('cuda')
# license_plate_detector.to('cuda')
# license_plate_recognition.to('cuda')
# car_brand_detector.to('cuda')

# Pre Model
run_model = "./pre_model/run_model.png"
car_model.predict(run_model)
license_plate_recognition.predict(run_model)
license_plate_detector.predict(run_model)
car_brand_detector.predict(run_model)

# Class_name_brand
class_names = {
    0: "Audi",
    1: "BMW",
    2: "BYD",
    3: "Benz",
    4: "Chevrolet",
    5: "Ford",
    6: "GWMORA",
    7: "Honda",
    8: "Isuzu",
    9: "MG",
    10: "Mazda",
    11: "Mini",
    12: "Mitsubishi",
    13: "Nissan",
    14: "Proton Exora",
    15: "Suzuki",
    16: "Toyota",
    17: "Toyota Alphard",
}


def clear_file(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)


def clear_folder(folder_path):
    shutil.rmtree(folder_path)


clear_file("./capture_car_to_pred")
clear_file("./capture_license_to_pred")
clear_file("./save_car")
clear_file("./save_license_plate")


def imgwrite(img):
    global stamp_day, stamp_time
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    stamp_day = now.strftime("%d/%m/%Y")
    stamp_time = now.strftime("%H:%M:%S")
    filename_car = "car_%s.jpeg" % current_time
    cv2.imwrite(os.path.join(input_test, filename_car), img)
    cv2.imwrite(os.path.join(input_car, filename_car), img)
    Path_car = input_car + "/" + filename_car
    return Path_car


def save_license_plate(img):
    now_license = datetime.now()
    current_license = now_license.strftime("%d_%m_%Y_%H_%M_%S")
    filename_license = "license_plate_%s.jpeg" % current_license
    cv2.imwrite(os.path.join(input_license_plate, filename_license), img)
    Path_plate = input_license_plate + "/" + filename_license
    return Path_plate


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        # print(colorsBGR)


def process_model(input_files):
    # load models
    global car_model

    # save license plate
    output_folder = "./capture_license_to_pred"

    # delete file before start
    file_list = os.listdir(output_folder)
    for file_name in file_list:
        file_path = os.path.join(output_folder, file_name)
        os.remove(file_path)

    # test detector license
    input_test = "./capture_car_to_pred"
    input_files = os.listdir(input_test)

    for filename in input_files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(input_test, filename)
            image = cv2.imread(file_path)

            # load video / images
            cap_model = cv2.VideoCapture(file_path)
            results = {}
            mot_tracker = Sort()
            vehicles = [2, 3, 5, 7]

            # read file in folder save license plate
            existing_files = os.listdir(output_folder)
            frame_number = (
                max(
                    [
                        int(filename.split("_")[1].split(".")[0])
                        for filename in existing_files
                    ],
                    default=-1,
                )
                + 1
            )

            frame_nmr = -1
            ret = True
            while ret:
                frame_nmr += 1
                ret, frame = cap_model.read()
                if ret:
                    results[frame_nmr] = {}
                    # detect vehicles
                    detections = car_model(frame)[0]
                    detections_ = []
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            detections_.append([x1, y1, x2, y2, score])

                    # track vehicles
                    track_ids = mot_tracker.update(np.asarray(detections_))

                    # detect brand
                    car_brand = car_brand_detector(frame)[0]
                    brn = ""
                    brn_score = ""
                    if len(car_brand.boxes) > 0:
                        for brand in car_brand.boxes.data.tolist():
                            xb1, yb1, xb2, yb2, b_score, b_class_id = brand
                            if b_score > 0.5:
                                class_name = class_names.get(b_class_id)
                                # print("รุ่นของรถ :", class_name, "\nconf :", b_score)
                                brn = class_name
                                brn_score = b_score

                    # detect license plates
                    license_plates = license_plate_detector(frame)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        # assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                            license_plate, track_ids
                        )

                        if car_id != -1:
                            # crop license plate
                            license_plate_crop = frame[
                                int(y1) : int(y2), int(x1) : int(x2), :
                            ]

                            # Resize the license plate crop to your desired dimensions
                            desired_width = 300  # Desired width of the license plate
                            desired_height = 150  # Desired height of the license plate
                            license_plate_crop = cv2.resize(
                                license_plate_crop, (desired_width, desired_height)
                            )

                            gray_license_plate = cv2.cvtColor(
                                license_plate_crop, cv2.COLOR_BGR2GRAY
                            )
                            mblur = cv2.medianBlur(gray_license_plate, 5)
                            equalized_license_plate = cv2.equalizeHist(mblur)

                            gaussian_blur = cv2.GaussianBlur(
                                gray_license_plate, (7, 7), 2
                            )
                            sharp_plate = cv2.addWeighted(
                                gray_license_plate, 1.5, gaussian_blur, -0.5, 0
                            )

                            _, binary_license_plate = cv2.threshold(
                                gray_license_plate,
                                0,
                                255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                            )
                            contours, _ = cv2.findContours(
                                binary_license_plate,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )

                            th1 = cv2.adaptiveThreshold(
                                gray_license_plate,
                                255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,
                                33,
                                1,
                            )

                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)

                                # Approximate the contour to 4 points (assuming it's a rectangle)
                                epsilon = 0.05 * cv2.arcLength(largest_contour, True)
                                approximated_contour = cv2.approxPolyDP(
                                    largest_contour, epsilon, True
                                )

                                # Draw the approximated contour on a blank image (black background)
                                contour_image = np.zeros_like(binary_license_plate)
                                cv2.drawContours(
                                    contour_image,
                                    [approximated_contour],
                                    -1,
                                    255,
                                    thickness=cv2.FILLED,
                                )

                                # Find the orientation angle of the license plate
                                angle = cv2.minAreaRect(approximated_contour)[-1]
                                # print(angle)

                                # Rotate the binary image to deskew the license plate
                                if angle < 360:
                                    if angle > 100 or angle < -100:
                                        angle = 86
                                    # Instead of fixed -90 degrees rotation, adjust the angle accordingly
                                    rotated_angle = angle - 90
                                    # Check if the angle is less than -45 after adjustment, and rotate accordingly
                                    if rotated_angle < -15:
                                        rotated_angle = 0

                                    angle = rotated_angle
                                # print(angle)

                                rotation_matrix = cv2.getRotationMatrix2D(
                                    tuple(
                                        np.array(binary_license_plate.shape[1::-1]) / 2
                                    ),
                                    angle,
                                    1,
                                )
                                deskewed_license_plate = cv2.warpAffine(
                                    gray_license_plate,
                                    rotation_matrix,
                                    gray_license_plate.shape[1::-1],
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                )

                            cv2.imwrite(
                                os.path.join(output_folder, filename),
                                deskewed_license_plate,
                            )

                            results_list = []
                            pred_files = [
                                f for f in os.listdir("capture_license_to_pred")
                            ]
                            for p_file in pred_files:
                                # Process each image
                                recognition_output = license_plate_recognition.predict(
                                    source=os.path.join(
                                        "capture_license_to_pred", p_file
                                    ),
                                    conf=0.4,
                                    save=True,
                                )

                                result = recognition_output[0]
                                box = result.boxes[0]
                                objects = []
                                for box in result.boxes:
                                    class_id = result.names[box.cls[0].item()]
                                    cords = box.xyxy[0].tolist()
                                    cords = [round(x) for x in cords]
                                    conf = round(box.conf[0].item(), 2)

                                    # Store data in a list
                                    objects.append(
                                        {
                                            "class_id": class_id,
                                            "coordinates": cords,
                                            "probability": conf,
                                        }
                                    )

                                # Sort objects by x-coordinate
                                sorted_objects = sorted(
                                    objects, key=lambda x: x["coordinates"][0]
                                )

                                # Generate the result string
                                day = f"วันที่ : {stamp_day}"
                                time = f"เวลา : {stamp_time}"
                                license_plate = ""
                                province = ""
                                brand_car = f"รุ่นของรถยนต์ : {brn}, conf : {brn_score}"

                                for obj in sorted_objects:
                                    if len(obj["class_id"]) > 3:
                                        province += obj["class_id"]
                                    else:
                                        license_plate += obj["class_id"]

                                # Append the result to the results list
                                results_list.append(
                                    f"รูป {p_file} \n{day} \n{time} \n{license_plate} \n{province} \n{brand_car}\n"
                                )
                                # cv2.imshow('Detected Car ROI', roi)
                                # cv2.imshow('crop', license_plate_crop)
                                Path_plate = save_license_plate(license_plate_crop)

                                # Save Json
                                # try:
                                #     with open(json_file_path, 'r') as json_file:
                                #         output_data = json.load(json_file)
                                # except FileNotFoundError:
                                #     output_data = {"license_plate_info": []}

                                # Set data to Save Json
                                minio_path_car = Path_car.split("/")[-1]
                                minio_path_plate = Path_plate.split("/")[-1]
                                new_data = {
                                    "date": stamp_day,
                                    "time": stamp_time,
                                    "license_plate": license_plate,
                                    "province": province,
                                    "brand": brn,
                                    "img_car_path": "images/cars/" + minio_path_car,
                                    "img_license_path": "images/licenses/"
                                    + minio_path_plate,
                                    # "./save_car/19_02_2024_21_43_31.png",
                                    # ./save_license_plate/19_02_2024_21_43_32.png"
                                    # images/cars/name_img
                                    # images/licenses/name_img
                                }

                                # output_data["license_plate_info"].append(new_data)
                                # with open(json_file_path, 'w') as json_file:
                                #     json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    for result in results_list:
        print(result)
    clear_folder("runs/detect")


# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
rtsp_url = "rtsp://192.168.1.209:1200/live"
# cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture('Demo_Video_Jettrack_25fps_real.mp4')

videoCapture = VideoCapture(source="rtsp://localhost:554/live", sizes=(1024, 576))
fps = FPS(1)

car_file_writer = FileWriter("car")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()
area = [(70, 385), (70, 510), (950, 510), (950, 385)]  # เปลี่ยนเป็นเส้นตรง
area_c = set()

videoCapture.start()

def runner():
    frame = videoCapture.read()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or videoCapture.is_stop():
        videoCapture.stop()
        return FPS_KEY.STOP
    
    car_file_writer.write_jpg(prefix="awdaw", image=frame)
    cv2.imwrite(os.path.join("./capture_car_to_pred2", "car_daw.jpeg"), frame)

    


    results = car_model.predict(frame)
    car_list = []
    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2, _, d = map(int, box)
            c = class_list[d]
            if "car" in c or "truck" in c:
                car_list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(car_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area, np.int32), ((x4, y4)), False)
        # Check if enough time has passed since the last count
        # if time.time() - last_count_time >= 3.5:
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
        if results >= 0:
            Path_car = imgwrite(frame)
            roi_size = 20  # Adjust the size of the ROI as needed
            roi = frame[
                max(0, y3 - roi_size) : min(frame.shape[0], y4 + roi_size),
                max(0, x3 - roi_size) : min(frame.shape[1], x4 + roi_size),
            ]

            area_c.add(id)
            # Update the last count time
            last_count_time = time.time()
            # process_model(os.listdir(input_test))
            # clear_file("./capture_car_to_pred")

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 69, 0), 2)
    # print(area_c)
    k = len(area_c)
    cv2.putText(frame, str(k), (90, 150), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 3)
    cv2.imshow("Video", frame)
    return FPS_KEY.CONTINUE

fps.run_with_non_block_fps(runner)

