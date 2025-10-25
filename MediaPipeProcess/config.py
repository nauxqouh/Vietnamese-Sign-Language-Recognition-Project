N_HAND_LANDMARKS = 21                                       # Using all hand landmarks (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
UPPER_BODY_CONNECTIONS = [                                  # Using upper body landmarks only without hand landmarks (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 23, 24
] 
N_POSE_LANDMARKS = len(UPPER_BODY_CONNECTIONS)
N_LANDMARKS = N_POSE_LANDMARKS + N_HAND_LANDMARKS*2         # Total number of landmarks for upper body and two hands (left + right)
K = 20
EPSILON_BODY=0.005
EPSILON_HAND_FINGER=0.003
EPSILON_EYE=0.001