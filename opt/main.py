import pickle
import module.face_embedding as embedding
import module.face_detection as face_detection
def main(path):
    # モデルのオープン
    with open('model.pickle', mode='rb') as f:
        modle = pickle.load(f)
    faces = face_detection.detect_faces(path)
    if faces:
        face = faces[0]
        # 顔ベクトルの取得
        face_vector = embedding.face_embedding_val(face,path)
        face_vector = face_vector["face_vector"]
        # 予測
        result=modle.predict([face_vector])
        if result[0] == 0:
            print("cute")
        else:
            print("normal")
        print(result)
    else:
        print("No faces detected.")

if __name__ == "__main__":
    file_path="image.png"
    main(file_path)