from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes

import uuid
import cv2
from pathlib import Path


def save_detection_results(results: Boxes) -> list[str]:
    """
    Save detection results as images if detections were found.

    :param results: Detection results from YOLO model prediction, containing bounding boxes and other metadata
    :return: List of paths where annotated images were saved as strings
    """
    # Initialize empty list to store paths of saved images
    saved_paths = []

    # Iterate through each detection result
    for i, result in enumerate(results):
        # Check if any detections were made by looking at number of bounding boxes
        if len(result.boxes) > 0:
            # Plot the detection results with bounding boxes and labels on the image
            annotated_image = result.plot()

            # Generate unique filename using UUID to avoid overwrites
            output_path = f"./run/img_{uuid.uuid4()}.jpg"

            # Save the annotated image to disk using OpenCV
            cv2.imwrite(output_path, annotated_image)

            # Get absolute path and convert to string for consistency
            saved_path = Path(output_path).resolve()
            print(f"Image saved to {saved_path}")
            saved_paths.append(str(saved_path))

    return saved_paths

def main():
    model = YOLOWorld('yolov8s-world.pt')
    model.set_classes(['human'])  # Set the classes you want to detect
    results: Boxes = model.predict('data/bus.jpg')
    save_detection_results(results)

if __name__ == "__main__":
    main()

