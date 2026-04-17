import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

def create_embeddings(dataset_path="dataset"):
    known_embeddings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                face = rgb[y:y+h, x:x+w]

                face = cv2.resize(face, (160, 160))
                face = np.expand_dims(face, axis=0)

                embedding = embedder.embeddings(face)[0]

                known_embeddings.append(embedding)
                known_names.append(person)

    print("✅ Embeddings Created")
    return known_embeddings, known_names