# qcar_recta_eventos.py — Line-by-line explanation

Written for: someone who needs to fully understand this script without modifying it.
Line numbers refer to `qcar_recta_eventos.py`. Related lines are grouped, but every line is covered.

## 1. Big picture

The script drives a Quanser **QCar** in a straight line and reacts to road events seen by the camera.

- **Main thread = perception**: camera frame -> YOLOv5n (ONNX, COCO classes) -> color validation / color fallback detectors -> event state machines -> a speed **factor** (0.0 to 1.0).
- **Control thread (50 Hz)** (only with `--drive`): holds heading with the gyroscope (P controller), applies the speed factor with ramped throttle, performs 90-degree turns when a giraffe is seen, and stops if perception stops updating.
- The two threads talk through the `Shared` object (lock-protected).

Run modes:

| Command | Effect |
|---|---|
| `python3 qcar_recta_eventos.py --probe` | Prints speed and gyro, car does not move |
| `python3 qcar_recta_eventos.py` | Perception only, car does not move |
| `python3 qcar_recta_eventos.py --drive` | Perception + motion (car moves) |
| `--no-display` | Extra flag: no video window |

Events:

| Event | Trigger | Result |
|---|---|---|
| STOP_APPROACH | Stop sign far (small box) | Half speed |
| STOP_SIGN | Stop sign close (bigger box) | Full stop 3 s, then continue (6 s cooldown) |
| PERSON | Person anywhere, big enough | Stop while visible |
| GIRAFFE_TURN | Giraffe seen | One 90-degree turn, then straight again |
| RED_LIGHT / YELLOW_LIGHT | Disabled (commented out) | — |

## 2. Lines 1–48: Docstring and imports

| Lines | Meaning |
|---|---|
| 1–29 | Module docstring (Spanish): usage, threads, active events, and the 4 tricks used to see the STOP sign early (crop to ROI so the sign looks bigger, CLAHE contrast, red-color validation of low-confidence YOLO boxes, color+shape fallback detector). |
| 31 | `argparse`: command-line flags. |
| 32 | `csv`: writes the logs at the end. |
| 33 | `math`: angles (radians/degrees, pi). |
| 34 | `os`: file paths. |
| 35 | `threading`: control thread, lock, stop event. |
| 36 | `time`: timing, sleeps, timestamps. |
| 37 | `traceback`: prints exception details in the control thread. |
| 39 | `cv2` (OpenCV): image processing and display. |
| 40 | `numpy`: array math. |
| 41 | `onnxruntime as ort`: runs the YOLO ONNX model. |
| 43 | `CameraProcessor` from the local `camera.py` (that is why the script must be in the same folder). |
| 45–48 | Tries to import Quanser's `QCar`; if the Quanser API is missing, `QCar = None` (checked later in `open_vehicle`). This lets perception-only mode run without the hardware library. |

## 3. Lines 50–97: Configuration constants

**General / model / camera**

| Line | Meaning |
|---|---|
| 51 | `MODEL_PATH`: location of `yolov5n.onnx` on the QCar. |
| 52 | `CAMERA_ID = "3"`: which camera to open. |
| 53 | Camera resolution 1640x820 at 30 fps. |
| 54 | `PRE_CONF = 0.25`: minimum YOLO confidence kept at all; finer filtering is done later. |
| 55 | `NMS_THRESHOLD = 0.45`: IoU above which overlapping boxes of the same class are merged. |

**Stop sign detection**

| Line | Meaning |
|---|---|
| 58 | `ENHANCE`: apply CLAHE contrast enhancement to YOLO input. |
| 59 | `STOP_CONF_HIGH = 0.50`: YOLO confidence at/above this -> accepted directly. |
| 60 | `STOP_CONF_LOW = 0.25`: between LOW and HIGH -> accepted only if the box is red enough. |
| 61 | `STOP_RED_MIN = 0.20`: minimum fraction of red pixels in the box. |
| 62 | `ENABLE_COLOR_DETECTOR`: turns on the red-octagon fallback detector. |
| 63 | `COLOR_MIN_AREA = 500`: smallest red blob (pixels) the fallback accepts. |
| 64 | Minimum saturation (100) and value/brightness (60) for a pixel to count as "red". |

**Giraffe detection and turn**

| Line | Meaning |
|---|---|
| 67 | `GIRAFFE_CONF_HIGH = 0.45`: accept directly. |
| 68 | `GIRAFFE_CONF_LOW = 0.25`: accept if enough yellow. |
| 69 | `GIRAFFE_YELLOW_MIN = 0.15`: minimum fraction of yellow pixels in the box. |
| 70 | `GIRAFFE_MIN_H = 0.05`: min box height / image height (ignores tiny/far giraffes). |
| 71 | `GIRAFFE_COOLDOWN = 8.0`: seconds before another turn can trigger (avoids re-triggering on the same giraffe). |
| 72 | `ENABLE_YELLOW_DETECTOR`: color fallback (yellow blob with brown spots). |
| 73 | `YELLOW_MIN_AREA = 2000`: smallest yellow blob (pixels). |
| 74 | `YELLOW_H = (15, 35)`: yellow hue range (OpenCV hue is 0–179). |
| 75 | Minimum saturation 80 and brightness 110 for yellow. |
| 76 | `BROWN_H = (3, 22)`: hue range of the giraffe's brown spots. |
| 77 | Brown: min saturation 70, brightness range 30–110. |
| 78 | `TURN_DIR = 1`: +1 = left, -1 = right (depends on the calibrated `STEER_SIGN`). |
| 79 | `TURN_ANGLE`: 90 degrees in radians. |
| 80 | `TURN_TOL`: 5 degrees; the turn is "done" when heading error is below this. |

**Vehicle / control**

| Line | Meaning |
|---|---|
| 82 | `VEHICLE_IO_RATE = 200`: QCar hardware I/O frequency (Hz). |
| 83 | `CONTROL_RATE = 50`: control loop frequency (Hz). |
| 84 | `STEER_MAX = 0.5`: steering saturation (rad). |
| 85 | `KP_HEADING = 1.0`: proportional gain, steering rad per rad of heading error. |
| 86 | `STEER_SIGN`: flip to -1.0 if the car corrects toward the wrong side. |
| 87 | `THROTTLE_CRUISE = 0.05`: normal throttle. |
| 88 | `THROTTLE_MIN = 0.035`: below this the motor cannot overcome friction, so nonzero commands are raised to this. |
| 89 | `ACCEL_RATE = 0.10`: throttle increase per second (gentle start). |
| 90 | `BRAKE_RATE = 1.00`: throttle decrease per second (almost immediate braking). |
| 91 | `STALE_S = 0.7`: if perception hasn't updated for this long -> safety stop. |
| 92 | `MAX_RUN_S = None`: optional time cap; `None` = unlimited (stop with Ctrl+C). |
| 95 | COCO class ids: person 0, traffic light 9, stop sign 11, giraffe 23. |
| 97 | `LOG_DIR`: folder of this script; CSV logs are saved there. |

## 4. Lines 100–170: `RoadEvent` class

A small state machine per event. It turns "detections this frame" into a speed factor.

**Docstring (102–110)**: three action types:
- `timed_stop`: stop for `duration` s, then `cooldown` s of grace.
- `hold`: apply `factor` while the object stays in the ROI.
- `trigger`: does not change speed; sets `fired` for one frame (used to request a turn), then waits `cooldown`.
- ROI = region of interest as image fractions `(x0, y0, x1, y1)`; `min_h` = minimum box height relative to image height (proxy for closeness); `color` is only for traffic lights.

**`__init__` (112–120)** stores the parameters. State variables:
- `active`: event currently on.
- `count`: consecutive frames with a hit (persistence).
- `clear`: consecutive frames without a hit.
- `t_end`: when a timed stop ends.
- `t_cool`: time before which the event cannot re-trigger.
- `fired`: one-frame flag for `trigger` events.

**`_set` (122–125)**: sets `active`, appends `(time, name, "ON"/"OFF")` to the log and prints it.

**`update` (127–170)**, called once per frame:
- 128–129: image height/width and the ROI.
- 130: `hit = False`.
- 131–140: loop over detections. Skip if wrong class or confidence below `min_conf` (132–133); skip if a color is required and doesn't match (134–135). Otherwise compute box center (136–137); if the center is inside the ROI **and** box height ratio >= `min_h` (138), it is a hit (139) and the detection is marked `used` (140), which the drawing code shows in green.
- 141–149 (`trigger`): reset `fired` (142); update the consecutive-hit counter (143); if hits >= `persist` and cooldown has passed (144), set `fired`, reset counter, start cooldown (145–146), log and print FIRE (147–148). Returns 1.0, i.e. never changes speed (149).
- 150–160 (`timed_stop`): if active and time is up -> turn off and start cooldown (151–154). If not active: count consecutive hits (156); when persistent and cooldown passed -> turn on, set end time, reset counter (157–160).
- 161–169 (`hold`): on a hit, increase `count` and reset `clear`; on a miss, reset `count` and increase `clear` (162–165). Turns on after `persist` hits (166–167); turns off after `clear_frames` frames with no hit (168–169).
- 170: return `factor` if active, else 1.0.

## 5. Lines 173–193: `make_events`

- 174: `stop_roi` = whole image.
- 175: commented traffic-light ROI.
- 178–179 **STOP_APPROACH**: class STOP_SIGN, `hold`, needs 2 consecutive frames, `min_h = 0.04` (far sign), `factor = 0.5` (half speed), releases after 8 empty frames.
- 180–181 **STOP_SIGN**: `timed_stop`, 3 frames, `min_h = 0.10` (close sign), stops 3 s, cooldown 6 s.
- 183–184 **PERSON**: whole image, confidence 0.50, 2 frames, `hold`, `min_h = 0.25`, `factor = 0` (full stop) while visible, releases after 5 empty frames.
- 186–187 **GIRAFFE_TURN**: whole image, `trigger` action, 3 frames, min height and cooldown from the config.
- 189–192: RED_LIGHT / YELLOW_LIGHT, commented out.

## 6. Lines 197–220: `Shared` class (thread communication)

| Lines | Meaning |
|---|---|
| 199 | Mutex lock. |
| 200 | `factor` (speed multiplier), `stamp` (time of last update), `active` (names of active events). |
| 201 | `ready`: perception is initialized (camera open, model warmed up). |
| 202 | `stop`: `threading.Event` that tells both threads to end. |
| 203 | `telemetry`: dictionary the control thread fills for the on-screen display. |
| 204 | `turn_requests`: counter of turn requests. |
| 206–208 | `request_turn`: increments the counter under the lock. |
| 210–212 | `get_turn_requests`: reads it. |
| 214–216 | `set_events`: stores factor, current time, and active list. |
| 218–220 | `get_events`: returns factor, **age** of the data (now - stamp), and a copy of the active list. Age is used for the stale-data safety stop. |

## 7. Lines 224–303: Perception helpers

**`load_model` (224–231)**: creates the ONNX session on CPU (225). Reads the input tensor's shape (226–227). Takes input height/width from the shape, falling back to 640 if dynamic (228–229). Returns the session and `meta = (input name, width, height, output names)` (230–231).

**`nms_indices` (234–257)**: Non-Maximum Suppression written in NumPy.
- 235–237: keep boxes with score >= threshold; return `[]` if none.
- 238: filter boxes and scores.
- 239–241: corners x1,y1 and x2,y2 from `[x, y, w, h]`, and box areas.
- 242: order by descending score.
- 243–256: loop: take the best box (245), save its original index (246), stop if it was the last (247–248). Compute intersection with the rest (249–254), IoU (255), and keep only boxes with IoU <= threshold (256).
- 257: return kept indices.

**`detect` (260–289)**: runs YOLO on one image.
- 261–262: unpack meta, image size.
- 263: scale `r` to fit the image in the model input while keeping aspect ratio (letterbox).
- 264–265: resized size and padding offsets `top`/`left`.
- 266–267: gray (114) canvas with the resized image centered.
- 268: BGR -> RGB (`::-1`), HWC -> CHW, to float32 in 0–1.
- 269: run the model (batch dimension added with `tensor[None]`).
- 270: flatten to rows `[cx, cy, w, h, objectness, class scores...]`.
- 271: drop rows with objectness < `PRE_CONF`.
- 272–273: return `[]` if nothing.
- 274–276: best class per row and final confidence = objectness x class score.
- 277–278: keep rows above threshold whose class is one we care about.
- 279–280: return `[]` if none.
- 281–283: convert center-format boxes to top-left `[x, y, w, h]` and undo padding/scaling to original crop coordinates.
- 284–287: NMS per class: shifting each class's boxes by `class_id * 10000` makes different classes never overlap, so e.g. a person cannot suppress a stop sign.
- 288–289: return a list of dicts `{cls, conf, box}`.

**`box_crop` (292–297)**: crops the frame to a box, clamped to image bounds.

**`union_roi` (300–302)**: smallest rectangle containing all event ROIs (used to crop the frame before YOLO).

## 8. Lines 305–426: Color tools

| Lines | Meaning |
|---|---|
| 305 | Creates a global CLAHE object (contrast limit 2.0, 8x8 tiles). |
| 308–310 `enhance` | Converts to LAB, applies CLAHE only to the lightness channel (keeps colors), converts back to BGR. |
| 313–316 `red_mask` | Converts to HSV; red is hue <= 10 or >= 165 (red wraps around the hue circle), with minimum saturation and brightness. Returns a boolean mask. |
| 319–323 `red_fraction` | Crops the box and returns the fraction of red pixels (0 if empty crop). |
| 326–330 `hsv_range` | Generic HSV range mask (hue range, min saturation, value range). |
| 333–334 `yellow_mask` | `hsv_range` with the yellow constants. |
| 337–338 `brown_mask` | `hsv_range` with the brown constants. |
| 341–345 `yellow_fraction` | Fraction of yellow pixels in a box. |
| 348–351 `giraffe_ok` | Accept if confidence >= HIGH; otherwise accept if >= LOW **and** enough yellow. |
| 354–372 `yellow_giraffes` | Fallback detector without YOLO. Builds yellow and brown masks (356), merges them (357), closes small gaps with a 9x9 kernel (358), finds outer contours (359; `[-2]` works across OpenCV versions). For each contour: skip if too small (362–363); compute bounding box and yellow/brown fractions (364–366); require >= 25% yellow and 5–45% brown, so single-color surfaces are rejected (368–369); add a detection with fixed confidence 0.50, box shifted back to full-frame coordinates (`ox`, `oy`), `src="color"` (370–371). |
| 375–378 `stop_ok` | Accept a stop sign if confidence >= HIGH, or >= LOW with red fraction >= `STOP_RED_MIN`. |
| 381–403 `red_octagons` | Fallback stop-sign detector: red mask (383), close gaps with 5x5 kernel (384), contours (385). Per contour, all must pass: area >= `COLOR_MIN_AREA` (389–390); width/height ratio 0.7–1.4, near square (392–393); convexity (area / convex-hull area) >= 0.85 (394–396); the blob fills >= 45% of its bounding box (397–398); polygon approximation has 6–10 sides, roughly an octagon (399–401). Passing blobs become detections with confidence 0.50 (402). |
| 406–409 `inside` | True if the center of detection `d` is inside any of the given boxes. Used to avoid duplicating a YOLO detection with a color one. |
| 412–426 `light_color` | (Only used by the disabled traffic light events.) Crops the light, returns "unknown" if too small (415–416). Counts "lit" pixels (saturation >= 90, brightness >= 130) whose hue is red, yellow or green (417–424), picks the color with most pixels (425), returns it if it covers >= 2% of the crop, else "unknown" (426). |

## 9. Lines 429–470: `draw` (on-screen overlay)

- 430: image size.
- 431–443: if `show_roi`, group events sharing the same ROI (432–434). For each group draw a rectangle, red and thick if any event is active, blue and thin otherwise (436–440), with a label listing event names (441–443). The `i % 2` offsets labels vertically so they do not overlap.
- 444: class id -> display name.
- 445–460: for each detection: integer box (446); green if used by an event, orange otherwise (447); label depends on source: color fallback shows `h/H` (449), YOLO shows confidence (451); append red fraction, yellow fraction, or light color if present (452–457); draw rectangle and text (458–460).
- 461–465: bottom-left text lines: speed factor and FPS; if telemetry exists, state/velocity/throttle and distance/heading/reference/steer (angles converted to degrees).
- 466–470: black background rectangle at the bottom, then draw each line in white.

## 10. Lines 473–537: `run_perception` (main thread loop)

| Lines | Meaning |
|---|---|
| 474–475 | Load model and print info. |
| 476 | Sorted set of class ids needed by the events (passed to `detect`). |
| 477 | Combined ROI for cropping. |
| 478 | Warm-up inference with zeros (first run is slow; avoids a startup spike). |
| 479 | Open the camera. |
| 480 | `shared.ready = True`: the control thread waits for this before moving. |
| 481 | Timers: start time, smoothed FPS, previous-frame time. |
| 482 | ROI display toggle. |
| 483–484 | `try` / loop until stop is requested. |
| 485–487 | Get a frame; if none, `continue` **without** updating the shared timestamp, so the control thread sees stale data and brakes on its own. |
| 488 | `now`: seconds since start. |
| 489–491 | Frame size, crop offsets `ox`, `oy`, and the cropped image. |
| 492 | Run YOLO on the crop (with CLAHE if `ENHANCE`). |
| 493–502 | For each detection: shift the box to full-frame coordinates (494–495), mark `src="yolo"` (496); stop signs get a red fraction (497–498); giraffes get a yellow fraction (499–500); traffic-light color is commented out (501–502). |
| 503 | Remove stop signs that fail `stop_ok`. |
| 504 | Remove giraffes that fail `giraffe_ok`. |
| 505–507 | If enabled, add color-based stop detections that do not overlap a YOLO stop. |
| 508–511 | Same for giraffes with the yellow detector. |
| 512 | Update every event with the detections; collect their speed factors. |
| 513–516 | `muted`: classes with a `timed_stop` active or in cooldown. Comment explains why: after the 3 s stop, STOP_APPROACH (half speed) must not immediately re-slow the car, so the car can resume. |
| 517–518 | `live`: keep an event's factor if it is a `timed_stop` itself, or its class is not muted. |
| 519 | Final factor = the minimum of all live factors (and 1.0); the most restrictive event wins. |
| 520 | Publish factor and active event names (also refreshes the timestamp). |
| 521–522 | If any event fired this frame (giraffe), request a turn. |
| 523–525 | Exponential moving average of FPS. |
| 526–533 | If display is on: draw overlay, show the window (527–528), read key (529); `q` or Esc quits (530–531); `r` toggles ROI drawing (532–533). |
| 534–537 | `finally`: always signal stop to the control thread, close camera, close windows. |

## 11. Lines 541–642: Vehicle control

**`open_vehicle` (541–547)**: raises an error if the Quanser API is missing (542–543); creates `QCar` in read mode 1 at 200 Hz (544); raises if the hardware card did not open, usually because another process holds it (545–546); returns the car.

**`read_sensors` (550–554)**: `read_write_std` sends throttle and steering and reads sensors in one call (551). Returns wheel speed from `motorTach` (552) and the yaw rate `wz` (z axis of the gyroscope) (553).

**`probe` (557–574)**: diagnostics only. Opens the car (558); prints a hint (560); 100 iterations sending zero commands (561–562); prints speed, `wz` and the whole gyro vector (563–566); if attributes are missing, lists what the car object has (567–570); waits 0.1 s (571). `finally` sends zeros and terminates the car (572–574).

**`wrap` (577–578)**: wraps an angle into [-pi, pi) so 359 degrees becomes -1 degree.

**`control_loop` (581–643)**, the 50 Hz thread:
- 582–584: open the car inside a `try`.
- 585: nominal period (0.02 s).
- 586–591: gyro calibration. With the car still, average 100 samples of `wz` to get the **bias** (drift offset).
- 592–593: wait until perception is ready (or stop is set).
- 594: initial heading `th`, reference heading `th_ref`, distance `dist`, throttle `thr` = 0.
- 595: `turning` flag and count of turn requests already handled.
- 596: start and previous times.
- 597: main loop until stop.
- 598–599: time step `dt` from real elapsed time.
- 600–602: optional time cap.
- 603–611: turn handling. If the shared request counter is higher than what was handled (606), mark it handled (607); if not already turning (608), set `turning`, move the reference heading by +/-90 degrees (610), and print (611). The same heading controller then performs the turn and continues straight along the new heading.
- 612: heading error, wrapped.
- 613–615: if turning and error < 5 degrees, the turn is done.
- 616: steering = sign x Kp x error, clipped to +/-`STEER_MAX` (the P controller).
- 617: read speed factor, its age, and active events.
- 618–621: if data is older than `STALE_S`, force factor 0 and state `SAFE_STOP`; else the state is the joined names of active events or `CRUISE`.
- 622–623: prefix `GIRO` while turning.
- 624–626: target throttle = cruise x factor; if nonzero, raise it to at least `THROTTLE_MIN`.
- 627–628: slew-rate limit: use `BRAKE_RATE` if decreasing, `ACCEL_RATE` if increasing; the throttle moves toward the target by at most `rate x dt` (smooth start, quick stop).
- 629: send commands and read sensors.
- 630: integrate heading: `th += (wz - bias) * dt`, wrapped.
- 631: integrate distance: `dist += |v| * dt`.
- 632–633: publish telemetry for the overlay.
- 634: append a row to the control log.
- 635: sleep the rest of the period to hold 50 Hz.
- 636–637: any exception prints a traceback.
- 638–643: `finally`: send zero throttle/steering, terminate the car, print message (639–642); always set stop so perception also ends (643).

## 12. Lines 647–695: Main

**`dump_csv` (647–655)**: skips if no rows (648–649); builds a path in `LOG_DIR` (650); writes header and rows (651–654); prints the saved path (655).

**`main` (658–691)**:
- 659–663: arguments `--probe`, `--drive`, `--no-display`.
- 664–666: `--probe` runs `probe()` and exits.
- 667–668: error if the model file is missing.
- 669: create `Shared` and the event list.
- 670: empty lists for the event log and control log.
- 671–677: with `--drive`, print a warning and start `control_loop` as a **daemon** thread (dies with the program); otherwise print that the car will not move.
- 678–680: run perception in the main thread; Ctrl+C is caught and reported.
- 682–691: `finally`: set stop, wait up to 5 s for the control thread (683–685), build a timestamp (686), and save two CSVs:
  - `events_<timestamp>.csv`: `t_s, evento, estado` (ON/OFF/FIRE).
  - `run_<timestamp>.csv`: `t_s, rumbo, rumbo_ref, v, dist, steer, throttle, estado, factor`.

**694–695**: standard entry point, calls `main()`.

## 13. Safety mechanisms (summary)

1. Stale perception (no update for 0.7 s) -> `SAFE_STOP` (factor 0).
2. Missing camera frames do not refresh the timestamp -> the same stop.
3. Any control-thread exception or exit -> `finally` sends zero commands and stops the car.
4. Ctrl+C, `q` or Esc -> shared stop event -> both threads end.
5. Throttle is ramped: gentle acceleration, rapid braking.

## 14. Things worth knowing when tuning

- `STEER_SIGN` and `TURN_DIR` must be calibrated together on the real car (line 78 and 86).
- `THROTTLE_MIN` (0.035) is the friction threshold; below it the car would not move.
- `min_h` values (0.04 far, 0.10 near) set the distances at which the stop events fire.
- Heading comes only from integrating the gyro after bias calibration, so it drifts slowly over long runs.
