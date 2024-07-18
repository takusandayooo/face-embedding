from keras_facenet import FaceNet
import cv2
from tf_explain.core.grad_cam import GradCAM
import numpy as np

embedder = FaceNet() # FaceNetモデル
def face_embedding(img,path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    embedding = embedder.embeddings([img_rgb]) # 潜在変数表現に変換
    return {"path":path,"face_vector":embedding[0]} # 顔ベクトルと画像のパスを返す
 
def face_embedding_val(img,path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    embedding = embedder.embeddings([img_rgb]) # 潜在変数表現に変換

    # 
    model = embedder.model # FaceNetのモデル（KerasのModelクラス）
    model.summary()

    # 前処理
    img = cv2.resize(img, (160, 160))
    X = np.float32([embedder._normalize(img)])
    data = (X, None)

    # 出力の重要要素を決定
    abs = np.abs(embedding) # 絶対値
    top_channel = np.argmax(abs) # 絶対値が最も大きい要素番号

    # GradCAMで可視化
    explainer = GradCAM()
    grid = explainer.explain(data, model, class_index=top_channel, layer_name="Block8_6_Conv2d_1x1")
    explainer.save(grid, ".", "grad_cam.png")
    return {"path":path,"face_vector":embedding[0]} # 顔ベクトルと画像のパスを返す

if __name__ == "__main__":
    img = cv2.imread('./01.jpg') # 画像の読み込み
    print(face_embedding(img,'./01.jpg')) # 顔ベクトルの取得


