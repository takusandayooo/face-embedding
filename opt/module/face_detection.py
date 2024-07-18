import cv2

def detect_faces(image_path, padding=0):
    # 画像を読み込み、グレースケールに変換
    img = cv2.imread(image_path)

    # 顔を検出
    face_detector = cv2.FaceDetectorYN_create("yunet_n_320_320.onnx", "", (320, 320), 0.6, 0.3, 5000, cv2.dnn.DNN_BACKEND_DEFAULT, target_id=cv2.dnn.DNN_TARGET_CPU)

    # 画像サイズを設定する
    face_detector.setInputSize((img.shape[1], img.shape[0]))

    # 顔検出
    _, faces = face_detector.detect(img)
    # 検出された顔の画像をリストに格納
    face_images = []
    if faces is None or len(faces) == 0:
        return False
    for face in faces:
        x, y, w, h = face[0:4]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 余白を追加
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        face_img = img[y:y+h, x:x+w]
        face_images.append(face_img)
    return face_images

if __name__ == "__main__":
    # 画像のパスを指定して顔検出を実行
    faces = detect_faces('./AKB48.jpg')
    if faces:
        for i, face in enumerate(faces):
            cv2.imwrite(f'face_{i}.jpg', face)
        print("Faces detected!")
    else:
        print("No faces detected.")
