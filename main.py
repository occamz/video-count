from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("input/test_count_1.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Define region points as a polygon covering the whole screen
region_points = [(0, 0), (w, 0), (w, h), (0, h)]

# Video writer
video_writer = cv2.VideoWriter(
    "output/result.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

classes_to_count = [
    0,  # Person
    # 1,  # Bicycle
    # 2,  # Car
    # 3,  # Motorcycle
    # 5,  # Bus
    # 6,  # Train
    # 7,  # Truck
    # ...
]

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

# NOTE: Skip frames, quicker but less accurate
FRAME_SKIP = 0

frame_number = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

    if FRAME_SKIP:
        frame_number += FRAME_SKIP
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


with open("output/result.txt", "w") as f:
    f.write(f"In: {counter.in_counts}\n" f"Out: {counter.out_counts}\n")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
