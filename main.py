import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Filter list
filters = [None, 'GRAYSCALE', 'SEPIA', 'NEGATIVE', 'BLUR']
current_filter = 0

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Debounce
last_action_time = 0
debounce_time = 1  # seconds

# Filter logic
def apply_filter(frame, filter_type):
    if filter_type == 'GRAYSCALE':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'SEPIA':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        return np.clip(sepia_frame, 0, 255).astype(np.uint8)
    elif filter_type == 'NEGATIVE':
        return cv2.bitwise_not(frame)
    elif filter_type == 'BLUR':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

while True:
    success, img = cap.read()
    if not success:
        print("Error reading frame.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get tip coordinates
            tips = {
                'thumb': hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                'index': hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                'middle': hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                'ring': hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                'pinky': hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
            }

            h, w, _ = img.shape
            tip_coords = {k: (int(v.x * w), int(v.y * h)) for k, v in tips.items()}

            for _, (x, y) in tip_coords.items():
                cv2.circle(img, (x, y), 10, (0, 255, 255), cv2.FILLED)

            now = time.time()

            # Click Picture
            if abs(tip_coords['thumb'][0] - tip_coords['index'][0]) < 30 and \
               abs(tip_coords['thumb'][1] - tip_coords['index'][1]) < 30:
                if now - last_action_time > debounce_time:
                    cv2.putText(img, "Picture Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_action_time = now
                    filename = f"picture_{int(now)}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"Picture saved as {filename}")

            # Change Filter
            elif any(abs(tip_coords['thumb'][0] - tip_coords[f][0]) < 30 and 
                     abs(tip_coords['thumb'][1] - tip_coords[f][1]) < 30 for f in ['middle', 'ring', 'pinky']):
                if now - last_action_time > debounce_time:
                    current_filter = (current_filter + 1) % len(filters)
                    last_action_time = now
                    print(f"Switched to filter: {filters[current_filter]}")

    # Apply Filter
    filtered_img = apply_filter(img, filters[current_filter])
    if filters[current_filter] == 'GRAYSCALE':
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Gesture-Controlled Photo App", filtered_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()