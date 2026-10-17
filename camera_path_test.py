
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

print("Loading segmentation model...")
processor = SegformerImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL)
model.eval()

# Cityscapes labels: road = 0, sidewalk = 1
ROAD = 0
SIDEWALK = 1

def connected_ground_mask(mask):
    surface = np.uint8((mask == ROAD) | (mask == SIDEWALK))

    # Find connected regions
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        surface, connectivity=8
    )

    h, w = surface.shape

    # Look for regions touching the bottom-center of the image
    bottom = labels[int(h * 0.95):, int(w * 0.35):int(w * 0.65)]

    valid_ids = set(np.unique(bottom))
    valid_ids.discard(0)

    result = np.zeros_like(surface, dtype=bool)

    for region_id in valid_ids:
        result |= labels == region_id

    return result

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("Camera started. Press Q to quit.")

frame_count = 0
mask = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # Run AI every 5 frames to reduce CPU load
        if frame_count % 5 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(
                images=Image.fromarray(rgb),
                return_tensors="pt"
            )

            with torch.inference_mode():
                outputs = model(**inputs)

                logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )

                mask = logits.argmax(dim=1)[0].cpu().numpy()

        display = frame.copy()

        if mask is not None:
            overlay = np.zeros_like(frame)

            # Green = road, blue = sidewalk
            overlay[mask == ROAD] = (0, 255, 0)
            overlay[mask == SIDEWALK] = (255, 0, 0)

            valid = (mask == ROAD) | (mask == SIDEWALK)
            blended = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
            display[valid] = blended[valid]

            # Divide the bottom half into three regions
            ground = connected_ground_mask(mask)

            lower = ground[h // 2:, :]
            sections = np.array_split(lower, 3, axis=1)

            sidewalk_scores = [
                float(np.mean(section == SIDEWALK))
                for section in sections
            ]

            road_scores = [
                float(np.mean(section == ROAD))
                for section in sections
            ]

            # Compare the amount of visible road and sidewalk
            scores = [
                float(np.mean(section))
                for section in sections
            ]


            names = ["LEFT", "CENTER", "RIGHT"]

            best_index = int(np.argmax(scores))
            best_direction = names[best_index]

            if scores[best_index] < 0.10:
                best_direction = "UNCLEAR"

            # Temporary GPS instruction for testing
            gps_direction = "LEFT"

            if best_direction == "UNCLEAR":
                navigation_result = "CAMERA UNCLEAR"
            elif best_direction == gps_direction:
                navigation_result = "VISUAL AREA MATCH"
            else:
                navigation_result = "DIRECTION NOT CONFIRMED"

            # Print results every 5 frames
            if frame_count % 5 == 0:
                print(
                    f"Sidewalk: {[round(x, 2) for x in sidewalk_scores]} | "
                    f"Road: {[round(x, 2) for x in road_scores]} | "
                    f"Result: {best_direction}",
                    flush=True
                )

            cv2.putText(
                display,
                f"Visible surface: {best_direction}",
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"GPS: {gps_direction} | {navigation_result}",
                (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            for i, (name, score) in enumerate(zip(names, scores)):
                x = int((i + 0.1) * w / 3)
                cv2.putText(
                    display,
                    f"{name}: {score:.0%}",
                    (x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        cv2.imshow("Camera Path Detection", display)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
