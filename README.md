# 顔の特徴量による分類

### 仕組み
FaceNetを用いて顔をベクトル化し、SVCで学習を行い写真の分類を行っている。

## 使い方
### 1. 環境構築
1. Dockerのダウンロード
   
1. コンテナの作成
``` 
docker compose up -d --build
```
2. コンテナに入る
```
docker compose exec python3 bash
```

### 2. 学習
1. 学習写真の設置  
[`cute`](./opt/photo/cute/),[`normal`](./opt/photo/normal/)というフォルダーに写真を設置してください  
※ 拡張子は`.jpg`,`.jpeg`,`.png`の写真ファイルに対応しています

2. [`opt`](./opt/)のディレクトリに入る
```
cd opt
```
3. [`train.py`](./opt/train.py)を実行 → モデルの作成
```
python train.py
```

### 3. 判定
1. optのディレクトリ内に`model.pickle`が作成されていることを確認
2. [`main.py`](./opt/main.py)を実行
   0,1のclassになっています。  
   ※関数の呼び出しのところで判定する写真ファイル名を記述してください


## 参考URL
[【超初心者向け】DockerでPythonの環境を構築する](https://qiita.com/_taketeru/items/1d547e95539d858b29a1)  

[FaceNetの顔認証をお手軽に試す](https://qiita.com/Takuya-Shuto-engineer/items/4dcbadbd16e16c3b1677)
