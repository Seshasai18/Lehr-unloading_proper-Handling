import cv2
import numpy as np
import os

BASE_VIDEO = r"D:\Resolute Office Works\Extracted_Frames\Daataset\unloading_annotated.mp4"

DASH_IMAGE = r"D:\Resolute Office Works\Bottle_count\ResoluteAI Dashboard.jpg"

OUTPUT_VIDEO = r"D:\Resolute Office Works\Bottle_count\final_dashboard_output.mp4"


cap = cv2.VideoCapture(BASE_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{BASE_VIDEO}")

dashboard = cv2.imread(DASH_IMAGE)

if dashboard is None:
    raise RuntimeError(f"Could not open dashboard image:\n{DASH_IMAGE}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("Input video:")
print(f"Width  : {w}")
print(f"Height : {h}")
print(f"FPS    : {fps:.2f}")
print(f"Frames : {n}")

PANEL_WIDTH = 400

output_width = w + PANEL_WIDTH
output_height = h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (output_width, output_height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video:\n{OUTPUT_VIDEO}"
    )         

frame_number = 0

while True:

    ok, frame = cap.read()

    if not ok:
        break

    frame_number += 1

    cleaned = frame.copy()

    top_h = min(370, h)

    cleaned[:top_h] = cv2.GaussianBlur(
        cleaned[:top_h],
        (101, 101),
        0
    )
    x1 = max(0, min(557, w - 1))
    x2 = max(x1 + 1, min(565, w - 1))

    left_x = max(0, x1 - 1)
    right_x = min(w - 1, x2)

    left = cleaned[:, left_x:left_x + 1]
    right = cleaned[:, right_x:right_x + 1]

    for x in range(x1, x2):

        alpha = (x - x1 + 1) / (x2 - x1 + 1)

        cleaned[:, x:x + 1] = (
            left.astype(np.float32) * (1 - alpha)
            + right.astype(np.float32) * alpha
        ).astype(np.uint8)

    panel = dashboard.copy()

    dashboard_h = panel.shape[0]

    footer_start = int(dashboard_h * 0.68)

    panel[footer_start:, :] = 255

    panel = cv2.resize(
        panel,
        (PANEL_WIDTH, h),
        interpolation=cv2.INTER_AREA
    )

    combined = np.hstack(
        (cleaned, panel)
    )

    writer.write(combined)

    if frame_number % 100 == 0:

        print(
            f"Processed {frame_number}/{n} frames "
            f"({frame_number / n * 100:.1f}%)"
        )

cap.release()
writer.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("FINAL OUTPUT CREATED")
print("=" * 60)

print(f"Output : {OUTPUT_VIDEO}")
print(f"Resolution : {output_width} x {output_height}")
print(f"FPS : {fps:.2f}")
print(f"Frames : {frame_number}")
print("=" * 60)