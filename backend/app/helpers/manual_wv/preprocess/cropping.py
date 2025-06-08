import os
import cv2

# Paths
DATASET_PATH = "../data/original"
OUTPUT_PATH = "../data/cropped"

# Traverse dataset
for writer_folder in sorted(os.listdir(DATASET_PATH)):
    writer_path = os.path.join(DATASET_PATH, writer_folder)

    if os.path.isdir(writer_path) and writer_folder.startswith("W"):
        writer_id = int(writer_folder.split("W")[1])
        if writer_id < 71:
            continue
        print(f"\nProcessing {writer_folder}...\n")
        images = sorted(os.listdir(writer_path))

        for idx, img_file in enumerate(images):
            img_path = os.path.join(writer_path, img_file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Skipping {img_file} (could not load)")
                continue

            sample_num = idx + 1
            writer_id = writer_folder
            print(f"Opening: {img_file} for {writer_folder}")

            cv2.namedWindow("Select Normal Speed Text", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Select Normal Speed Text", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow("Select Normal Speed Text", image)
            roi1 = cv2.selectROI("Select Normal Speed Text", image, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Normal Speed Text")

            cv2.namedWindow("Select Fast Speed Text", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Select Fast Speed Text", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow("Select Fast Speed Text", image)
            roi2 = cv2.selectROI("Select Fast Speed Text", image, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Fast Speed Text")

            # Extract crops
            for i, roi in enumerate([roi1, roi2]):
                x, y, w, h = roi
                cropped_img = image[y:y+h, x:x+w]
                label = "N" if i == 0 else "F"
                filename = f"{writer_id}_S{sample_num:02d}_{label}.png"
                print("Writer ID:", writer_id)
                output_writer_folder = os.path.join(OUTPUT_PATH, writer_id)
                print("Writer Folder", output_writer_folder)
                os.makedirs(output_writer_folder, exist_ok=True)
                cv2.imwrite(os.path.join(output_writer_folder, filename), cropped_img)
                print(f"Saved: {filename}")

            cv2.destroyAllWindows()
            