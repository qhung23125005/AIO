import cv2
from ultralytics import solutions

def main():
    cap = cv2.VideoCapture("data/highway.mp4")
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)

    region_points = [
        (430, 700),
        (1600, 700),
        (1600, 1080),
        (430, 1080)
    ] # Count the objects inside this region

    video_writer = cv2.VideoWriter(
        "run/highway_counted.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model='yolo11x.pt'
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with the object counter
        results = counter(frame)

        # Write the frame to the output video
        video_writer.write(results.plot_im)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
