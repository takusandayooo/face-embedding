import module.face_detection as face_detection
import module.face_embedding as face_embedding
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


def train_main():
    dir_cute="./photo/cute/"
    dir_normal="./photo/normal/"
    face_vectors = []
    face_cute_or_normal_txt = []
    # 画像のパスを指定して顔検出を実行
    files_cute = os.listdir(dir_cute)
    files_normal = os.listdir(dir_normal)
    for file in files_cute:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(dir_cute, file)
            faces = face_detection.detect_faces(file_path)
            if faces:
                face = faces[0]
                # 顔ベクトルの取得
                face_vector = face_embedding.face_embedding(face,file)
                face_vectors.append(face_vector["face_vector"])
                face_cute_or_normal_txt.append(0)
            else:
                print("No faces detected.")

    for file in files_normal:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(dir_normal, file)
            faces = face_detection.detect_faces(file_path)
            if faces:
                face = faces[0]
                # 顔ベクトルの取得
                face_vector = face_embedding.face_embedding(face,file)
                face_vectors.append(face_vector["face_vector"])
                face_cute_or_normal_txt.append(1)
            else:
                print("No faces detected.")
    
    print("face_vectors length = ", len(face_vectors), "face_cute_or_normal_txt length = ", len(face_cute_or_normal_txt))
    x_train, x_test, y_train, y_test = train_test_split(face_vectors, face_cute_or_normal_txt, test_size = 0.2, train_size = 0.8, shuffle = True)   
    print("train length = ", len(x_train), "test length = ", len(x_test))

    model_svc=SVC()
    model_svc.fit(x_train,y_train)

    y_pred = model_svc.predict(x_test)
    print("正解率 = " , accuracy_score(y_test, y_pred))

    with open('model.pickle', mode='wb') as f:
        pickle.dump(model_svc,f,protocol=2)

if __name__ == "__main__":
    train_main()