import cv2

def detect_faces(image_path):
    # 顔検出のためのカスケード分類器をロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 画像を読み込み、グレースケールに変換
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 検出された顔の画像をリストに格納
    face_images = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_images.append(face_img)

    if len(face_images) == 0:
        return False

    return face_images
if __name__ == "__main__":
    # 画像のパスを指定して顔検出を実行
    faces = detect_faces('./01.jpg')
    if faces:
        cv2.imwrite('face.jpg', faces[0])
        print("Faces detected!")
    else:
        print("No faces detected.")
