from keras_facenet import FaceNet
import cv2

embedder = FaceNet() # FaceNetモデル
def face_embedding(img,path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    embedding = embedder.embeddings([img_rgb]) # 潜在変数表現に変換
    return {"path":path,"face_vector":embedding[0]} # 顔ベクトルと画像のパスを返す
 
if __name__ == "__main__":
    img = cv2.imread('./01.jpg') # 画像の読み込み
    print(face_embedding(img,'./01.jpg')) # 顔ベクトルの取得