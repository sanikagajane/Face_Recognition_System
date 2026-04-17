import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

def recognize_faces(known_embeddings, known_names, image_path="group.jpg"):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)
    attendance = []

    for face_data in faces:
        x, y, w, h = face_data['box']
        face = rgb[y:y+h, x:x+w]

        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)

        embedding = embedder.embeddings(face)[0]

        # Compare
        min_dist = float("inf")
        name = "Unknown"

        for i, known_embedding in enumerate(known_embeddings):
            dist = np.linalg.norm(embedding - known_embedding)

            if dist < min_dist:
                min_dist = dist
                name = known_names[i]

        if min_dist > 0.9:
            name = "Unknown"

        attendance.append(name)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    print("👥 Detected:", attendance)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return attendance