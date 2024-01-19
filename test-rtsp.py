import cv2
# RTSP URL
rtsp_url = "rtsp://192.168.1.101:1200/live"
#rtsp_url = 'rtsp://10.66.13.130:1200/live'
# cap = cv2.VideoCapture(rtsp_url)

target_resolution = (1200, 750)
# Set the target resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])

# Create a VideoCapture object
# target_frame_rate = 15
# cap.set(cv2.CAP_PROP_FPS, target_frame_rate)
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Loop to continuously read frames
while True:
    # Read a frame from the RTSP stream
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Display the frame
    cv2.imshow("RTSP Stream", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()