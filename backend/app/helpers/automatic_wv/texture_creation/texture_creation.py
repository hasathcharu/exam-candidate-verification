import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def segment_lines(image, threshold=350000, line_height=180, size=1600):
    projection = np.sum(image, axis=1) #horizontal projection
    line_start = None
    lines = []

    for i, value in enumerate(projection):
        if value < threshold and line_start is None:
            line_start = i
        elif value >= threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None

    cropped_lines = []
    for (start, end) in lines:
        if end - start < 20:
            continue
        center = (start + end) // 2
        bottom = max(0, center - line_height // 2)
        top = min(image.shape[0], center + line_height // 2)
        cropped_lines.append(cv2.resize(image[bottom:top, :], (size, line_height)))
    return clean_lines(cropped_lines)

def clean_lines(dirty_lines):
    cleaned_lines = []
    for line in dirty_lines:
        contours, _ = cv2.findContours(
            cv2.bitwise_not(line), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros_like(line)
        min_area = 40
        for contour in contours:
            middle_point = False
            for point in contour:
                if point[0][1] > 40 and point[0][1] < 140:
                    middle_point = True
                    break
            if cv2.contourArea(contour) > min_area and middle_point:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        cleaned_binary = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(line), mask))
        cleaned_lines.append(cleaned_binary)
    return cleaned_lines, dirty_lines

def segment_words(lines):
    cropped_words = []
    for line in lines:
        vertical_projection = np.sum(line, axis=0)
        word_threshold = 45000
        word_start = None
        words = []
        consecutive_count = 0
        min_consecutive = 15
        for i, value in enumerate(vertical_projection):
            if value < word_threshold:
                consecutive_count = 0
                if word_start is None:
                    word_start = i
            elif value >= word_threshold:
                if word_start is not None:
                    consecutive_count += 1
                    if consecutive_count >= min_consecutive:
                        words.append((word_start, i - min_consecutive + 1))
                        word_start = None
                        consecutive_count = 0
        for word in words:
            start, end = word
            cropped_words.append(line[:, start:end])
    return cropped_words

def generate_texture(cropped_words, file_name="unknown"):
    canvas_height = 950
    canvas_width = 950
    if not cropped_words:
        print(f"[WARNING] No cropped words for file: {file_name}")
        return np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    row_spacing = -100
    column_spacing = 0

    rows = []
    current_row = []
    current_width = 0

    for word in cropped_words:
        word_height, word_width = word.shape

        while word_width > 0:
            remaining_space = canvas_width - current_width

            if remaining_space >= word_width:
                current_row.append(word)
                current_width += word_width + column_spacing
                break
            else:
                if remaining_space > 0:
                    word_part = word[:, :remaining_space]
                    current_row.append(word_part)
                    word = word[:, remaining_space:]
                    word_width = word.shape[1]
                rows.append(current_row)
                current_row = []
                current_width = 0

    if not rows:
        print(f"[WARNING] No rows could be constructed from the cropped words in file: {file_name}")
        return np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    if current_row:
        rows.append(current_row)

    used_width = sum(w.shape[1] for w in rows[-1]) + column_spacing * (len(rows[-1]) - 1)
    remaining_space = canvas_width - used_width
    for w in rows[0]:
      if remaining_space <= 0:
          break
      h, w_w = w.shape
      if w_w + column_spacing <= remaining_space:
          rows[-1].append(w)
          remaining_space -= (w_w + column_spacing)
      else:
          slice_width = remaining_space
          rows[-1].append(w[:, :slice_width])
          remaining_space = 0

    texture = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    y_offset = 0
    row_index = 0

    while y_offset < canvas_height:
        row = rows[row_index % len(rows)]  # Loop through rows cyclically
        x_offset = 0
        max_row_height = max(word.shape[0] for word in row)

        if y_offset + max_row_height > canvas_height:
            break

        for word in row:
            word_height, word_width = word.shape
            if y_offset + word_height <= canvas_height and x_offset + word_width <= canvas_width:
                word_text = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
                word_text[y_offset:y_offset + word_height, x_offset:x_offset + word_width] = word
                texture = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_not(texture), cv2.bitwise_not(word_text)))
            x_offset += word_width + column_spacing

        y_offset += max_row_height + row_spacing
        row_index += 1

    crop_size = 900

    h, w = texture.shape
    y_start = (h - crop_size) // 2
    x_start = (w - crop_size) // 2
    y_end   = y_start + crop_size
    x_end   = x_start + crop_size

    texture = texture[
        y_start : y_end,
        x_start : x_end
    ]

    return texture

def split_texture(texture):
    patch_size = 450
    patches = []

    for i in range(2):  # Rows
        for j in range(2):  # Columns
            y_start = i * patch_size
            x_start = j * patch_size
            patch = texture[y_start:y_start + patch_size, x_start:x_start + patch_size]
            patches.append(patch)
    return patches

def create_texture(image, output_dir=None, prefix="image"):
    cleaned_lines, _ = segment_lines(image)
    cropped_words = segment_words(cleaned_lines)
    texture = generate_texture(cropped_words)
    patches = split_texture(texture)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for idx, patch in enumerate(patches):
            filename = f"{prefix}_T{idx+1}.png"
            path = os.path.join(output_dir, filename)
            plt.imsave(path, patch, cmap="gray")

    return patches
