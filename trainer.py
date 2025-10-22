import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import Counter

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_exercise_from_pose(landmarks):
    """
    Robust exercise detection using both arms and multiple body landmarks.
    Returns 'press', 'curl', or 'unknown'.
    """
    mp_pose = mp.solutions.pose
    
    try:
        # Get landmarks for BOTH arms
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate angles for both arms
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_angle = (l_angle + r_angle) / 2
        
        # Average Y positions (remember: smaller Y = higher on screen)
        avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        avg_elbow_y = (l_elbow[1] + r_elbow[1]) / 2
        
        # SHOULDER PRESS Detection:
        # Wrists are AT or ABOVE shoulder level (smaller Y value)
        # Elbows bent around 80-120 degrees
        # Wrists are above or at elbow level
        wrists_at_shoulder_height = (avg_wrist_y <= avg_shoulder_y + 0.08)
        elbows_bent = (75 < avg_angle < 125)
        wrists_above_elbows = (avg_wrist_y <= avg_elbow_y + 0.05)
        
        if wrists_at_shoulder_height and elbows_bent and wrists_above_elbows:
            return "press"
        
        # BICEP CURL Detection:
        # Arms mostly extended downward (angle > 140)
        # Wrists are BELOW elbows (larger Y value)
        # Wrists are BELOW shoulders
        arms_extended = (avg_angle > 140)
        wrists_below_elbows = (avg_wrist_y > avg_elbow_y + 0.03)
        wrists_below_shoulders = (avg_wrist_y > avg_shoulder_y + 0.05)
        
        if arms_extended and wrists_below_elbows and wrists_below_shoulders:
            return "curl"
            
    except Exception as e:
        pass
    
    return "unknown"

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- Video File Path ---
    video_path = r"C:\Users\karth\OneDrive\Desktop\RepRight AI trainer\8401288-sd_360_640_30fps.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    # --- Output Video Setup ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter(f'workout_analysis_{timestamp}.mp4', fourcc, fps, (frame_width, frame_height))

    # --- State Variables ---
    exercise_type = None
    rep_counter = 0
    state = "DETECTING"
    feedback = ""
    detection_samples = []
    DETECTION_SAMPLES_NEEDED = 15
    debug_mode = False
    
    print("\n" + "="*50)
    print("  RepRight AI Trainer - Starting...")
    print("="*50)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug view")
    print("  'c' - Manually set to CURL")
    print("  'p' - Manually set to PRESS")
    print("\nDetecting exercise...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- AUTO DETECTION PHASE ---
            if exercise_type is None:
                detected = detect_exercise_from_pose(landmarks)
                if detected != "unknown":
                    detection_samples.append(detected)
                
                if len(detection_samples) >= DETECTION_SAMPLES_NEEDED:
                    most_common = Counter(detection_samples).most_common(1)[0][0]
                    exercise_type = most_common
                    state = "GET_READY"
                    print(f"\n✓ AUTO-DETECTED: {exercise_type.upper()}")
                    print(f"  (Press 'c' or 'p' to override if incorrect)\n")

            # --- EXERCISE PROCESSING ---
            if exercise_type is not None:
                mp_pose_obj = mp.solutions.pose
                r_shoulder = [landmarks[mp_pose_obj.PoseLandmark.RIGHT_SHOULDER.value].x, 
                             landmarks[mp_pose_obj.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose_obj.PoseLandmark.RIGHT_ELBOW.value].x, 
                          landmarks[mp_pose_obj.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose_obj.PoseLandmark.RIGHT_WRIST.value].x, 
                          landmarks[mp_pose_obj.PoseLandmark.RIGHT_WRIST.value].y]
                elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                if exercise_type == 'press':
                    if state == "GET_READY" and 75 < elbow_angle < 125: 
                        state = "READY"
                    elif state == "READY" and elbow_angle > 160 and r_wrist[1] < r_shoulder[1]: 
                        state = "UP"
                    elif state == "UP":
                        feedback = "Good form" + "!" * min(rep_counter + 1, 5)
                        if 75 < elbow_angle < 125:
                            rep_counter += 1
                            state = "READY"
                            feedback = ""

                elif exercise_type == 'curl':
                    if state == "GET_READY" and elbow_angle > 150: 
                        state = "READY"
                    elif state == "READY" and elbow_angle < 40: 
                        state = "DOWN"
                    elif state == "DOWN":
                        feedback = "Good form" + "!" * min(rep_counter + 1, 5)
                        if elbow_angle > 150:
                            rep_counter += 1
                            state = "READY"
                            feedback = ""

        except:
            pass

        # --- UI DISPLAY ---
        cv2.rectangle(frame, (0, 0), (frame_width, 100), (245, 117, 16), -1)
        
        cv2.putText(frame, 'REPS', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(rep_counter), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.putText(frame, 'STATE', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, state, (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        if exercise_type:
            cv2.putText(frame, f'Exercise: {exercise_type.upper()}', (420, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Detection progress
        if exercise_type is None:
            progress = len(detection_samples) / DETECTION_SAMPLES_NEEDED
            bar_width = int(500 * progress)
            cv2.rectangle(frame, (20, frame_height - 50), (20 + bar_width, frame_height - 20), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, frame_height - 50), (520, frame_height - 20), (255, 255, 255), 2)
            cv2.putText(frame, 'AUTO-DETECTING EXERCISE...', (25, frame_height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'C' for Curl or 'P' for Press to skip", (25, frame_height - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Feedback
        if feedback:
            cv2.rectangle(frame, (0, frame_height - 90), (500, frame_height - 10), (0, 255, 0), -1)
            cv2.putText(frame, feedback, (20, frame_height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Debug info
        if debug_mode and exercise_type:
            cv2.putText(frame, f"Angle: {int(elbow_angle)}", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        out.write(frame)
        cv2.imshow('RepRight AI Trainer', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('c'):  # Manual override to CURL
            exercise_type = 'curl'
            state = "GET_READY"
            print("\n✓ MANUALLY SET TO: CURL")
        elif key == ord('p'):  # Manual override to PRESS
            exercise_type = 'press'
            state = "GET_READY"
            print("\n✓ MANUALLY SET TO: PRESS")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print(f"  Analysis Complete!")
    print("="*50)
    print(f"  Exercise: {exercise_type.upper() if exercise_type else 'N/A'}")
    print(f"  Total Reps: {rep_counter}")
    print(f"  Video saved: workout_analysis_{timestamp}.mp4")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
