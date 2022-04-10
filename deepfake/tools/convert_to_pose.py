import cv2
import os
import mediapipe as mp

if not os.path.exists("check"):
    os.mkdir("check")
if not os.path.exists("check/raw"):
    os.mkdir("check/raw")
if not os.path.exists("check/pro"):
    os.mkdir("check/pro")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture('ahjnxtiamx.mp4')
count = 0
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        raw_path = os.path.join("check/raw",str(count)+'.jpg')
        cv2.imwrite(raw_path, frame)

        result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_drawing.draw_landmarks(frame, result.face_landmarks,mp_holistic.FACEMESH_TESSELATION,\
                                  landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
        pro_path = os.path.join("check/pro",str(count)+'.jpg')
        cv2.imwrite(pro_path, frame)

        count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

