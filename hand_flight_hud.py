from left_hand_control import control_throttle, control_yaw
import cv2
import mediapipe as mp
import time
import os
import math
import numpy as np

# Suppress TensorFlow / MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def put_text_with_shadow(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                         font_scale=0.7, color=(0,255,0), thickness=2):
    x, y = org
    # shadow
    cv2.putText(frame, text, (x+2, y+2), font, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
    # main
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_bar(frame, top_left, width, height, value_pct, bar_color=(0,255,0)):
    """
    Draws a horizontal bar: value_pct in [0..100]
    top_left: (x,y)
    """
    x, y = top_left
    # background
    cv2.rectangle(frame, (x, y), (x+width, y+height), (50,50,50), -1)
    # filled portion
    filled = int((value_pct / 100.0) * width)
    cv2.rectangle(frame, (x, y), (x+filled, y+height), bar_color, -1)
    # border
    cv2.rectangle(frame, (x, y), (x+width, y+height), (200,200,200), 1)

def landmarks_to_commands(landmarks, frame_w, frame_h):
    """
    Given a single hand's landmarks (normalized), compute simple flight command values:
    - throttle: distance between thumb_tip (4) and index_tip (8), mapped 0..100
    - pitch: angle of wrist->index_tip in degrees (-90..90)
    - roll: angle of wrist->middle_finger_mcp in degrees (-90..90)
    - yaw: x offset of hand center vs frame center mapped to -100..100
    """
    # Convert normalized landmarks to pixel coords
    pts = [(int(l.x * frame_w), int(l.y * frame_h), l.z) for l in landmarks.landmark]
    # important landmark indices:
    # 0: wrist, 4: thumb_tip, 8: index_tip, 9: middle_finger_mcp
    wrist = np.array(pts[0][:2], dtype=np.float32)
    thumb_tip = np.array(pts[4][:2], dtype=np.float32)
    index_tip = np.array(pts[8][:2], dtype=np.float32)
    mid_mcp = np.array(pts[9][:2], dtype=np.float32)

    # throttle: distance thumb <-> index
    dist = np.linalg.norm(thumb_tip - index_tip)
    # Normalize by diagonal of frame
    diag = math.hypot(frame_w, frame_h)
    throttle = (dist / diag) * 200.0  # scale a bit
    throttle = max(0.0, min(100.0, throttle))

    # pitch: angle of vector wrist->index_tip relative to horizontal (degrees)
    v_idx = index_tip - wrist
    pitch = math.degrees(math.atan2(-v_idx[1], v_idx[0]))  # negative y for camera coords -> conventional
    # map to -90..90
    if pitch > 90: pitch = 90
    if pitch < -90: pitch = -90

    # roll: angle wrist->mid_mcp relative to vertical
    v_mid = mid_mcp - wrist
    roll = math.degrees(math.atan2(v_mid[1], v_mid[0]))
    # clamp to -90..90
    if roll > 90: roll = 90
    if roll < -90: roll = -90

    # yaw: horizontal offset of hand center from frame center -> -100..100
    hand_center_x = np.mean([p[0] for p in pts])
    yaw = ((hand_center_x - (frame_w/2)) / (frame_w/2)) * 100.0
    yaw = max(-100.0, min(100.0, yaw))

    return throttle, pitch, roll, yaw

def draw_flight_hud(frame, throttle, pitch, roll, yaw, fps=None):
    """
    Draw textual HUD using putText (with shadow) and small bars.
    """
    h, w = frame.shape[:2]
    # semi-transparent panel background
    panel_w, panel_h = 360, 160
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    alpha = 0.45
    # place panel top-right
    x0, y0 = w - panel_w - 10, 10
    overlay = frame.copy()
    overlay[y0:y0+panel_h, x0:x0+panel_w] = (
        overlay[y0:y0+panel_h, x0:x0+panel_w] * (1-alpha) + panel * alpha
    ).astype(np.uint8)
    frame[y0:y0+panel_h, x0:x0+panel_w] = overlay[y0:y0+panel_h, x0:x0+panel_w]

    # Title
    put_text_with_shadow(frame, "FLIGHT COMMANDS", (x0+12, y0+28), font_scale=0.7, color=(200,200,0))

    # Numerical values + bars
    put_text_with_shadow(frame, f"Throttle: {throttle:5.1f} %", (x0+12, y0+55), font_scale=0.6)
    draw_bar(frame, (x0+150, y0+42), 190, 12, throttle)

    put_text_with_shadow(frame, f"Pitch   : {pitch:6.1f}°", (x0+12, y0+85), font_scale=0.6)
    # convert pitch -90..90 to 0..100 for bar
    pitch_pct = ((pitch + 90.0) / 180.0) * 100.0
    draw_bar(frame, (x0+150, y0+72), 190, 12, pitch_pct, bar_color=(255,160,0))

    put_text_with_shadow(frame, f"Roll    : {roll:6.1f}°", (x0+12, y0+115), font_scale=0.6)
    roll_pct = ((roll + 90.0) / 180.0) * 100.0
    draw_bar(frame, (x0+150, y0+102), 190, 12, roll_pct, bar_color=(0,160,255))

    put_text_with_shadow(frame, f"Yaw     : {yaw:6.1f}", (x0+12, y0+145), font_scale=0.6)
    yaw_pct = ((yaw + 100.0) / 200.0) * 100.0
    draw_bar(frame, (x0+150, y0+132), 190, 12, yaw_pct, bar_color=(160,255,120))

    # FPS (optional)
    if fps is not None:
        put_text_with_shadow(frame, f"FPS: {int(fps)}", (10, 30), font_scale=0.8, color=(0,200,255))

def webcam_hand_tracking(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam is running... Press 'q' to quit. Press 's' to save screenshot.")

    prev_time = 0  # For FPS calculation

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Warning: Failed to capture frame. Exiting...")
                break

            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            result = hands.process(rgb_frame)

            # Default commands (zeroed)
            combined_throttle = 0.0
            combined_pitch = 0.0
            combined_roll = 0.0
            combined_yaw = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Identify hand type
                    label = hand_label.classification[0].label
                    wrist_x = int(hand_landmarks.landmark[0].x * frame_w)
                    wrist_y = int(hand_landmarks.landmark[0].y * frame_h)
                    put_text_with_shadow(frame, label, (wrist_x - 30, wrist_y - 30), font_scale = 0.7, color = (0,255,0))

                    # Extract raw gesture data
                    throttle, pitch, roll, yaw = landmarks_to_commands(hand_landmarks, frame_w, frame_h)

                    # Apply library only for left hand
                    if label =="Left":
                        combined_throttle = control_throttle(throttle)
                        combined_yaw = control_yaw(yaw)
                    elif label == "Right":
                        combined_pitch = pitch
                        combined_roll = roll

                    

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
            prev_time = curr_time

            # Draw the HUD with numeric values
            draw_flight_hud(frame, combined_throttle, combined_pitch, combined_roll, combined_yaw, fps=fps)

            # Show the frame
            cv2.imshow("Hand Tracking Webcam with Flight HUD", frame)

            key = cv2.waitKey(1) & 0xFF
            # Exit when 'q' is pressed
            if key == ord('q'):
                print("👋 Exiting...")
                break
            # Save screenshot if 's' is pressed
            elif key == ord('s'):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"📸 Screenshot saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    webcam_hand_tracking()
