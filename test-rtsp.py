import cv2
# RTSP URL
#rtsp_url = 'rtsp://192.168.1.101:1200/live'
rtsp_url = 'rtsp://192.168.1.226:1200/live'

cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Loop to continuously read frames
while True:
    ret, frame = cap.read() # Read a frame from the RTSP stream

    if not ret:
        print("Error: Failed to read frame.") # Check if the frame was successfully read
        break
    
    cv2.imshow("RTSP Stream", frame) # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()