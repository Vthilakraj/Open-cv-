import cv2
import mediapipe as mp
import pyautogui
import time

# --- Configuration ---
# Adjust these thresholds based on your hand size, camera distance, and personal preference
JUMP_THRESHOLD_Y = 0.05       # Upward movement threshold for jump (percentage of screen height)
SLIDE_THRESHOLD_Y = 0.05      # Downward movement threshold for slide (percentage of screen height)
LEFT_THRESHOLD_X = 0.05       # Leftward movement threshold (percentage of screen width)
RIGHT_THRESHOLD_X = 0.05      # Rightward movement threshold (percentage of screen width)

COOLDOWN_TIME = 0.3           # Cooldown time in seconds between actions to prevent rapid firing
MIN_HAND_SIZE_PX = 50         # Minimum bounding box size for hand to be considered (prevents noise)

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- PyAutoGUI Cooldown Variables ---
last_action_time = 0

# --- Previous Hand State for Movement Detection ---
prev_wrist_y = None
prev_wrist_x = None

# --- Webcam Initialization ---
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Point your hand at the camera.")
print("Gestures:")
print("  - Jump: Quick upward movement of your hand.")
print("  - Slide: Quick downward movement of your hand.")
print("  - Move Left: Quick leftward movement of your hand.")
print("  - Move Right: Quick rightward movement of your hand.")
print("\nPress 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get wrist landmark for overall hand position
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_wrist_x = wrist_landmark.x * width
            current_wrist_y = wrist_landmark.y * height

            # --- Calculate hand bounding box for size check ---
            x_coords = [lm.x * width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * height for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            hand_width = max_x - min_x
            hand_height = max_y - min_y

            # Only process gestures if hand is sufficiently large
            if hand_width < MIN_HAND_SIZE_PX or hand_height < MIN_HAND_SIZE_PX:
                # print("Hand too small, ignoring gestures.")
                prev_wrist_x = current_wrist_x
                prev_wrist_y = current_wrist_y
                continue # Skip gesture detection for small hands

            # --- Gesture Recognition Logic ---
            if current_time - last_action_time > COOLDOWN_TIME:
                if prev_wrist_y is not None and prev_wrist_x is not None:
                    # Jump detection (Upward movement)
                    if (prev_wrist_y - current_wrist_y) / height > JUMP_THRESHOLD_Y:
                        print("Gesture: JUMP!")
                        pyautogui.press('up')
                        last_action_time = current_time
                        # Reset prev_wrist to prevent re-triggering on same movement
                        prev_wrist_y = current_wrist_y
                        prev_wrist_x = current_wrist_x
                        continue # Action triggered, skip other checks

                    # Slide detection (Downward movement)
                    elif (current_wrist_y - prev_wrist_y) / height > SLIDE_THRESHOLD_Y:
                        print("Gesture: SLIDE!")
                        pyautogui.press('down')
                        last_action_time = current_time
                        prev_wrist_y = current_wrist_y
                        prev_wrist_x = current_wrist_x
                        continue # Action triggered, skip other checks

                    # Move Left detection (Leftward movement)
                    elif (prev_wrist_x - current_wrist_x) / width > LEFT_THRESHOLD_X:
                        print("Gesture: MOVE LEFT!")
                        pyautogui.press('left')
                        last_action_time = current_time
                        prev_wrist_y = current_wrist_y
                        prev_wrist_x = current_wrist_x
                        continue # Action triggered, skip other checks

                    # Move Right detection (Rightward movement)
                    elif (current_wrist_x - prev_wrist_x) / width > RIGHT_THRESHOLD_X:
                        print("Gesture: MOVE RIGHT!")
                        pyautogui.press('right')
                        last_action_time = current_time
                        prev_wrist_y = current_wrist_y
                        prev_wrist_x = current_wrist_x
                        continue # Action triggered, skip other checks

            # Update previous wrist coordinates for the next frame's comparison
            prev_wrist_x = current_wrist_x
            prev_wrist_y = current_wrist_y
    else:
        # If no hand is detected, reset previous position to avoid jump/slide on hand re-entry
        prev_wrist_x = None
        prev_wrist_y = None

    # Display the frame
    cv2.imshow('Subway Surfers Hand Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application stopped.")