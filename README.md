# Lane Keeping Assist (OpenCV)

A small Python program that finds the lane lines in a driving video, colours the lane and shows how far the car is from the lane centre.

![Example output](images/final.png)

## What it does

- Detects the left and right lane lines with classic OpenCV (no machine learning).
- Colours the area between the lanes.
- Shows the offset of the car from the lane centre as a percentage of the lane width. The text turns red above 10%.
- Keeps working through gaps in dashed lines.
- Stops the two lines from crossing on narrow views.

## How it works

For every frame:

1. Resize the frame to a fixed width.
2. Convert to grayscale.
3. Blur to remove noise.
4. Find edges (Canny).
5. Keep only a triangle in front of the car.
6. Find straight line segments (Hough transform).
7. Sort the segments into left and right lane by slope.
8. Fit one straight line per side.
9. Remember the last line of each side, so dashed lanes do not disappear.
10. Shorten the lines if they get closer than a minimum gap.
11. Colour the lane and calculate the offset.

A longer explanation with an image for every step is in [`docs/Lane_Keeping_Assist_Guide.pdf`](docs/Lane_Keeping_Assist_Guide.pdf).

## Requirements

- Python 3
- OpenCV and NumPy

```
pip install opencv-python numpy
```

## Usage

```
python main.py .\Data\test1.mp4
```

Press `q` to quit.

## Settings

| Setting | What it does |
|---|---|
| `WIDTH = 800` | Frame width in pixels. Lower it if the window is too big. |
| `SMOOTH = 0.8` | How much the old line is trusted. Higher is steadier, lower reacts faster. |
| `MAX_MISSED = 25` | Frames a lane is kept after it was last seen. |
| `MIN_GAP = 40` | Smallest gap in pixels allowed between the two lines. |
| `y_top = int(h * 0.6)` | How high up the lines are drawn (inside `process`). Lower the number for longer lines. |
| `abs(offset) > 10` | Offset in % where the text turns red (inside `process`). |

## Limitations

- Only straight lane lines are modelled. Sharp curves are not followed well.
- The camera is assumed to be in the middle of the car.
- The offset is a percentage of the lane width, not a distance in metres.
- The settings are tuned for a forward-facing dashcam view. Other footage will probably need changes.
- Shadows, worn paint, rain and night driving reduce detection quality.