import cv2
from ultralytics import YOLO
import os
import sys
import time

# Load the trained YOLO model
model = YOLO("path/to/your/custom/trained/model.pt")

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    # Run YOLO inference on the image
    results = model(image)
    # Visualize the results on the image
    annotated_image = results[0].plot()
    # Display the annotated image
    cv2.imshow("YOLOv11 Inference - Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLO inference on the frame
            results = model(frame)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Display the annotated frame
            cv2.imshow("YOLOv11 Inference - Video", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    # Open the default webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not access the webcam")
        return
    # Loop through the webcam frames
    while cap.isOpened():
        success, frame = cap.read()
        # time.sleep(2)
        if success:
            # Resize the frame to 640x640
            resized_frame = cv2.resize(frame, (1080, 720))
            # Run YOLO inference on the resized frame
            results = model(resized_frame)

            boxes = results[0].boxes

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 1 and conf > 0.6:
                    print("Fall Detected!")
                    # alert_system.send_alert("Fall Detected")

            
            # Display the annotated frame
            cv2.imshow("YOLOv11 Inference - Webcam", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # Release the webcam and close the display window
    cap.release()
    cv2.destroyAllWindows()


def main(input_source):
    if input_source.lower() == "webcam":
        process_webcam()
    elif os.path.isfile(input_source):
        # Check the file extension to determine if it's an image or video
        ext = os.path.splitext(input_source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            process_image(input_source)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            process_video(input_source)
        else:
            print(f"Error: Unsupported file extension {ext}")
    else:
        print(f"Error: {input_source} is not a valid file or 'webcam'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_source>")
        print("Where <input_source> is 'webcam' or the path to an image/video file")
    else:
        input_source = sys.argv[1]
        print(input_source)
        main(input_source)
