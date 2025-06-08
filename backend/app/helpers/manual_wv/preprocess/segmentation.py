import numpy as np
import cv2

def segment_lines(image, threshold=350000, line_height=180, size=1600):
    projection = np.sum(image, axis=1)
    line_start = None
    lines = []

    for i, value in enumerate(projection):
        if value < threshold and line_start is None:
            line_start = i
        elif value >= threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None

    cropped_lines = []
    cropped_rectangles = []
    for (start, end) in lines:
        if end - start < 20:
            continue
        center = (start + end) // 2
        bottom = max(0, center - line_height // 2)
        top = min(image.shape[0], center + line_height // 2)
        cropped_lines.append(cv2.resize(image[bottom:top, :], (size, line_height)))
        cropped_rectangles.append((0, bottom, image.shape[1], top))
    return clean_lines(cropped_lines), cropped_rectangles

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

def segment_words(lines, word_threshold=45000, min_consecutive=15):
    cropped_words = []
    for line in lines:
        vertical_projection = np.sum(line, axis=0)
        word_start = None
        words = []
        consecutive_count = 0
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
            cropped_words.append(line[:, max(start-5, 0):min(line.shape[1], end+5)])
    return cropped_words
