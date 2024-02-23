import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count the number of fingers
            finger_count = 0

            # Thumb (Landmarks 4, 8, 12, 16, 20)
            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                finger_count += 1
            # Index finger (Landmarks 8, 12, 16, 20)
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y:
                finger_count += 1
            # Middle finger (Landmarks 12, 16, 20)
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y:
                finger_count += 1
            # Ring finger (Landmarks 16, 20)
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                finger_count += 1
            # Pinky finger (Landmarks 20)
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y:
                finger_count += 1

            # Display the finger count for each hand
            cv2.putText(frame, f"Fingers ({results.multi_hand_landmarks.index(hand_landmarks) + 1}): {finger_count}",
                        (10, 30 * (results.multi_hand_landmarks.index(hand_landmarks) + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Finger Count", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
