# AutoSubJP

英語動画に自動で日本語字幕を生成して焼き込むツール

## 必要条件

- Python 3.8以上
- ffmpeg（システムにインストール済みであること）
- NVIDIA GPU（推奨）またはCPU

## インストール方法

1. リポジトリをクローン：
```bash
git clone https://github.com/yourusername/AutoSubJP.git
cd AutoSubJP
```

2. 仮想環境を作成して有効化：
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. 必要なパッケージをインストール：
```bash
pip install -r requirements.txt
```

4. Hugging Faceのトークンを設定：
   - https://huggingface.co/settings/tokens でトークンを生成
   - `.env`ファイルに以下の形式で記載：
   ```
   HUGGING_FACE_TOKEN=your_token_here
   ```

## 使用方法

### GUIモード（推奨）

エクスプローラーで入力動画と出力先を選択できます：
```bash
python auto_sub_jp.py --gui
# または
python auto_sub_jp.py -g
```

### コマンドラインモード

```bash
python auto_sub_jp.py --input 入力動画ファイル [--output 出力ファイル名]
# または
python auto_sub_jp.py -i 入力動画ファイル [-o 出力ファイル名]
```

例：
```bash
python auto_sub_jp.py -i input.mp4 -o output_with_subtitles.mp4
```

## 機能

- Whisperを使用した音声認識（baseモデル）
- staka/fugumt-en-jaモデルによる高精度な機械翻訳
- SRT形式の字幕ファイル生成
- ffmpegによる字幕の動画への焼き込み
- GUIによる簡単なファイル選択
- GPU/CPU自動検出と利用
- プログレスバーによる進捗表示

## モデルとキャッシュについて

### モデルの保存場所
- Whisperモデル: `~/.cache/whisper`（約1GB）
- 翻訳モデル: `~/.cache/huggingface/hub`（約2GB）

### キャッシュのクリア
不要になったモデルは以下のフォルダを削除することでクリアできます：
```bash
# Windows
%USERPROFILE%\.cache\whisper
%USERPROFILE%\.cache\huggingface

# Linux/Mac
~/.cache/whisper
~/.cache/huggingface
```

## GPU対応について

- NVIDIA GPUが利用可能な場合は自動的に使用されます（CUDA）
- Apple Silicon（M1/M2）の場合はMPSバックエンドを使用
- 上記以外の場合はCPUを使用

## 字幕スタイル

- フォント: Yu Gothic
- フォントサイズ: 24
- 文字色: 白
- 背景: 半透明の黒
- ボーダースタイル: 影付き

## 注意事項

- 入力動画は英語音声であることを前提としています
- 処理時間は動画の長さとハードウェアによって変動します
- 初回実行時は必要なモデルのダウンロードに時間がかかります（約3GB）
- Windows環境でのシンボリックリンクの警告は無視して問題ありません
- 長い動画の場合、十分なディスク容量があることを確認してください
