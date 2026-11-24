import sys
import cv2
import numpy as np

WIDTH = 800   # display width in pixels; lower it if the window is still too big
SMOOTH = 0.8        # 0 = trust only the new frame, closer to 1 = trust the old line more
MAX_MISSED = 30     # how many frames to keep a lane after it was last seen
MIN_GAP = 20        # minimum pixels between the two lane lines at the top (stops them crossing)

last = {"left": None, "right": None}     # last known line for each side
missed = {"left": 0, "right": 0}         # frames since each side was last seen


def lane_points(lines, y_bottom, y_top):
    """Average many line segments into ONE straight line, return its two end points."""
    pts_x, pts_y = [], []
    for x1, y1, x2, y2 in lines:
        pts_x += [x1, x2]
        pts_y += [y1, y2]
    # Fit x = a*y + b (using y as input avoids division by zero on vertical lines)
    a, b = np.polyfit(pts_y, pts_x, 1)
    return int(a * y_bottom + b), y_bottom, int(a * y_top + b), y_top


def process(frame):
    h, w = frame.shape[:2]

    # 1. Grayscale: colour is not needed to find bright lane paint.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. Blur: removes small noise so Canny does not detect random specks.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny: keeps only pixels where brightness changes sharply (edges).
    edges = cv2.Canny(blur, 50, 150)

    # 4. Region of interest: black out everything except a triangle in front of the car.
    mask = np.zeros_like(edges)
    triangle = np.array([[(0, h), (w // 2, int(h * 0.5)), (w, h)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    edges = cv2.bitwise_and(edges, mask)

    # 5. Hough transform: turns edge pixels into straight line segments.
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                               minLineLength=40, maxLineGap=100)

    # 6. Split segments into left / right lane by slope.
    #    In image coordinates (y points down) the left lane has negative slope.
    left, right = [], []
    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:          # ignore near-horizontal lines
                continue
            (left if slope < 0 else right).append((x1, y1, x2, y2))

    # 7. One averaged straight line per side, with memory.
    y_bottom, y_top = h, int(h * 0.65)
    for name, side in (("left", left), ("right", right)):
        if side:                                   # lane seen in this frame
            new = np.array(lane_points(side, y_bottom, y_top), float)
            old = last[name]
            # Smoothing: mostly keep the old line, nudge it toward the new one (less jitter).
            last[name] = new if old is None else SMOOTH * old + (1 - SMOOTH) * new
            missed[name] = 0
        else:                                      # lane NOT seen (e.g. gap in a dashed line)
            missed[name] += 1
            if missed[name] > MAX_MISSED:          # gone for too long: forget it
                last[name] = None
    lanes = [last[n].astype(int) for n in ("left", "right") if last[n] is not None]

    # 7b. Stop the two lines from touching/crossing where they converge (narrow views).
    #     If the gap between them at the top is smaller than MIN_GAP, shorten both lines
    #     to the height where the gap is exactly MIN_GAP.
    if len(lanes) == 2:
        (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = lanes
        gap_bottom, gap_top = rx1 - lx1, rx2 - lx2
        if gap_bottom <= MIN_GAP:                  # lines already meet at the bottom: bad detection
            lanes = []
        elif gap_top < MIN_GAP:
            t = (gap_bottom - MIN_GAP) / (gap_bottom - gap_top)   # 0 = bottom, 1 = old top
            new_y = int(ly1 + t * (ly2 - ly1))
            lanes = [np.array([lx1, ly1, int(lx1 + t * (lx2 - lx1)), new_y]),
                     np.array([rx1, ry1, int(rx1 + t * (rx2 - rx1)), new_y])]

    # 8. Colour the space between the lanes (only if both lanes were found).
    if len(lanes) == 2:
        (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = lanes
        overlay = frame.copy()
        area = np.array([[(lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)]], np.int32)
        cv2.fillPoly(overlay, area, (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)   # 30% green, 70% original

    for x1, y1, x2, y2 in lanes:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)

    # 9. Offset as a % of the lane width, measured at the MIDDLE of each lane line
    #    (the midpoint of a segment is halfway between its bottom and top end point).
    if len(lanes) == 2:
        left_x = (lanes[0][0] + lanes[0][2]) / 2
        right_x = (lanes[1][0] + lanes[1][2]) / 2
        mid_y = (lanes[0][1] + lanes[0][3]) // 2  # halfway up the visible part of the lanes
        lane_center = (left_x + right_x) / 2
        lane_width = right_x - left_x
        offset = (lane_center - w / 2) / lane_width * 100   # >0: lane centre is right of the car
        color = (0, 0, 255) if abs(offset) > 10 else (0, 255, 0)
        cv2.putText(frame, f"Offset: {offset:+.1f} %", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.line(frame, (w // 2, mid_y - 20), (w // 2, mid_y + 20), (255, 255, 255), 3)            # car centre
        cv2.line(frame, (int(lane_center), mid_y - 20), (int(lane_center), mid_y + 20), color, 3)  # lane centre
    return frame


if __name__ == "__main__":
    video_file = sys.argv[1] if len(sys.argv) > 1 else None
    if video_file is None:
        print("Usage: python main.py <video_file>")
        sys.exit(1)
    
    video = cv2.VideoCapture(video_file)
    while True:
        ok, frame = video.read()
        if not ok:
            break
        scale = WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        cv2.imshow("Lane Keeping Assist", process(frame))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()