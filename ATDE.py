import cv2
import numpy as np

def ATDE(video_path, main_colors, start_year, end_year,
                         hsv_range=10, lower_sv=100, upper_sv=255,
                         min_neighbors=5, output_path=None,
                         red_restriction=None, green_restriction=None, blue_restriction=None):
    """
    Extracts pixel counts representing territorial extent from an animated history video.

    Parameters:
        video_path (str): Path to the input video file.
        main_colors (list of lists): RGB colors indicating the territory.
        start_year (int): The starting year for the analysis.
        end_year (int): The ending year for the analysis.
        hsv_range (int): Hue range tolerance in HSV color space.
        lower_sv (int): Lower bound for saturation and value in HSV space.
        upper_sv (int): Upper bound for saturation and value in HSV space.
        min_neighbors (int): Minimum neighbors for direct neighbor filtering.
        output_path (str, optional): Path to save the validation video. If None, validation video is not created.
        red_restriction (str, optional): Restriction on red channel in format ('<', value) or ('>', value).
        green_restriction (str, optional): Restriction on green channel in format ('<', value) or ('>', value).
        blue_restriction (str, optional): Restriction on blue channel in format ('<', value) or ('>', value).

    Returns:
        dict: Dictionary mapping years to pixel counts.
    """
    def rgb_to_hsv(rgb):
        rgb_array = np.uint8([[rgb[::-1]]])
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
        return hsv[0][0]

    def direct_neighbor_filter(mask, min_neighbors):
        binary_mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbors = cv2.filter2D(binary_mask, -1, kernel)
        return np.where((neighbors >= min_neighbors) & (binary_mask == 1), 255, 0).astype(np.uint8)

    def parse_restriction(restriction):
        if restriction is None or restriction == '':
            return None
        operator = restriction[0]
        value = int(restriction[1:])
        if operator not in ['<', '>']:
            raise ValueError(f"Invalid operator: {operator}")
        return operator, value

    # Inverted interpretation to match intended “exclude extreme values” semantics.
    def apply_restriction(channel, restriction):
        if restriction is None:
            return np.ones(channel.shape, dtype=bool)
        operator, value = restriction
        if operator == '<':
            return channel >= value
        elif operator == '>':
            return channel <= value
        else:
            raise ValueError(f"Invalid operator: {operator}")

    # Parse restrictions
    red_restriction = parse_restriction(red_restriction)
    green_restriction = parse_restriction(green_restriction)
    blue_restriction = parse_restriction(blue_restriction)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_year = max(1, total_frames // (end_year - start_year + 1))

    # Convert main colors to HSV ranges
    hsv_colors = [rgb_to_hsv(c) for c in main_colors]
    lower_bounds = [np.array([h[0] - hsv_range, lower_sv, lower_sv]) for h in hsv_colors]
    upper_bounds = [np.array([h[0] + hsv_range, upper_sv, upper_sv]) for h in hsv_colors]

    pixel_counts = {}
    frame_idx, year_idx = 0, 0

    # Define highlight color as the mean of main_colors
    highlight_color = np.mean(main_colors, axis=0).astype(int)

    # Set up validation video writer
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out_vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Combine all HSV ranges
        for lower, upper in zip(lower_bounds, upper_bounds):
            mask |= cv2.inRange(hsv_frame, lower, upper)

        # Apply RGB restrictions
        b, g, r = cv2.split(frame)
        mask &= apply_restriction(r, red_restriction).astype(np.uint8) * 255
        mask &= apply_restriction(g, green_restriction).astype(np.uint8) * 255
        mask &= apply_restriction(b, blue_restriction).astype(np.uint8) * 255

        mask = direct_neighbor_filter(mask, min_neighbors)

        # Sample frame per year
        if frame_idx % frames_per_year == 0:
            year = start_year + year_idx
            pixel_counts[year] = int(np.count_nonzero(mask))
            year_idx += 1

        # Write validation frame
        if output_path:
            validation_frame = np.zeros_like(frame)
            validation_frame[mask > 0] = highlight_color
            out_vid.write(validation_frame)

        frame_idx += 1

    cap.release()
    if output_path:
        out_vid.release()

    return pixel_counts
