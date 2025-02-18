import os
import argparse
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import subprocess
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import warnings

def get_device():
    """
    利用可能なデバイス（GPU/CPU）を取得し、GPUの使用を強制
    """
    if torch.cuda.is_available():
        # CUDA設定を強制
        torch.cuda.set_device(0)  # 最初のGPUを使用
        print(f"NVIDIA GPU ({torch.cuda.get_device_name(0)})を使用します")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPUを使用します")
        return torch.device("mps")
    else:
        print("CPUを使用します")
        return torch.device("cpu")

def setup_huggingface():
    """
    Hugging Faceの認証設定を行う
    """
    # .envファイルから環境変数を読み込む
    load_dotenv()
    
    # トークンを取得
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        print("警告: HUGGING_FACE_TOKENが設定されていません。")
        print(".envファイルに以下の形式でトークンを設定してください：")
        print("HUGGING_FACE_TOKEN=your_token_here")
        return False
        
    try:
        # Hugging Faceにログイン
        login(token=token)
        return True
    except Exception as e:
        print(f"Hugging Faceログインエラー: {str(e)}")
        return False

def transcribe_audio(video_path, device):
    """
    Whisperを使用して動画から音声を文字起こしし、タイムスタンプ付きのセグメントを取得
    
    Args:
        video_path (str): 入力動画のパス
        device (torch.device): 利用するデバイス
    
    Returns:
        list: 文字起こしされたセグメントのリスト（各セグメントは開始時間、終了時間、テキストを含む）
    """
    print("音声認識を開始します...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        model = whisper.load_model("base").to(device)
        result = model.transcribe(video_path)
    
    return result["segments"]

def translate_text(text, device=None):
    """
    英語のテキストを日本語に翻訳
    
    Args:
        text (str): 翻訳する英語テキスト
        device (torch.device, optional): 使用するデバイス
    
    Returns:
        str: 翻訳された日本語テキスト
    """
    try:
        # モデルとトークナイザーの設定
        model_name = "staka/fugumt-en-ja"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # GPUが利用可能な場合は強制的に使用
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.to(device)
        
        # 入力テキストのエンコード
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 翻訳の実行
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                length_penalty=0.6,
                early_stopping=True
            )
        
        # 翻訳結果のデコード
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
        
    except Exception as e:
        return text

def format_time(seconds):
    """
    秒数をSRT形式のタイムスタンプに変換
    
    Args:
        seconds (float): 秒数
    
    Returns:
        str: "HH:MM:SS,mmm"形式の文字列
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = round(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt(segments, output_srt, device):
    """
    翻訳済みテキストとタイムスタンプからSRTファイルを生成
    
    Args:
        segments (list): 文字起こしセグメントのリスト
        output_srt (str): 出力するSRTファイルのパス
        device (torch.device): 利用するデバイス
    """
    print("字幕ファイルを生成しています...")
    
    # プログレスバーの設定（leave=Trueで最後の行を残し、dynamic_ncolsで1行に収める）
    progress = tqdm(total=len(segments), desc="字幕生成", unit="セグメント", leave=True, dynamic_ncols=True, position=0)
    
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            # 英語テキストを日本語に翻訳（引数のdeviceを利用）
            translated_text = translate_text(segment["text"], device)
            
            # タイムスタンプをフォーマット
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            
            # SRTフォーマットで書き込み
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{translated_text}\n\n")
            
            # プログレスバーを更新
            progress.update(1)
    
    progress.close()

def burn_subtitles(input_video, srt_path, output_video):
    """
    ffmpegを使用して字幕を動画に焼き込む
    
    Args:
        input_video (str): 入力動画のパス
        srt_path (str): SRTファイルのパス
        output_video (str): 出力動画のパス
    """
    print("字幕を動画に焼き込んでいます...")
    cmd = [
        "ffmpeg", "-i", input_video,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Yu Gothic,FontSize=24,PrimaryColour=&HFFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0'",
        "-c:a", "copy",
        "-y", output_video
    ]
    try:
        # ffmpegの出力を抑制
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"エラー: ffmpegの実行中にエラーが発生しました。\n{e.stderr}")
        raise

def select_input_video():
    """
    エクスプローラーを開いて入力動画を選択
    
    Returns:
        str: 選択された動画ファイルのパス、キャンセルされた場合はNone
    """
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示
    print("入力動画を選択してください...")
    file_path = filedialog.askopenfilename(
        title="入力動画を選択",
        filetypes=[
            ("動画ファイル", "*.mp4 *.avi *.mov *.mkv"),
            ("すべてのファイル", "*.*")
        ]
    )
    return file_path if file_path else None

def select_output_path(default_filename):
    """
    エクスプローラーを開いて出力ファイルのパスを選択
    
    Args:
        default_filename (str): デフォルトのファイル名
    
    Returns:
        str: 選択された出力パス、キャンセルされた場合はNone
    """
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示
    print("出力ファイルを選択してください...")
    file_path = filedialog.asksaveasfilename(
        title="出力ファイルを選択",
        defaultextension=".mp4",
        initialfile=default_filename,
        filetypes=[("MP4ファイル", "*.mp4")]
    )
    return file_path if file_path else None

def main():
    parser = argparse.ArgumentParser(description="英語動画に日本語字幕を自動生成して焼き込むツール")
    parser.add_argument("--input", "-i", help="入力動画ファイルのパス")
    parser.add_argument("--output", "-o", help="出力動画ファイルのパス")
    parser.add_argument("--gui", "-g", action="store_true", help="GUIモードを使用")
    args = parser.parse_args()

    # 初回実行時の注意メッセージ
    print("注意: 初回実行時は必要なモデルをダウンロードするため、時間がかかる場合があります。")

    # Hugging Face認証の設定
    if not setup_huggingface():
        if args.gui:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("エラー", "Hugging Faceの認証に失敗しました。\n.envファイルのトークンを確認してください。")
        return

    # GUIモードまたはコマンドライン引数が不足している場合、ファイル選択ダイアログを表示
    input_video = args.input
    output_video = args.output

    if args.gui or not input_video:
        input_video = select_input_video()
        if not input_video:
            print("入力ファイルが選択されませんでした。")
            return

    if args.gui or not output_video:
        default_output = "output_with_subtitles.mp4"
        output_video = select_output_path(default_output)
        if not output_video:
            print("出力ファイルが選択されませんでした。")
            return

    # 一時的なSRTファイルのパス
    temp_srt = "temp_subtitles.srt"

    # デバイス取得
    device = get_device()

    try:
        # 1. 音声認識 (deviceを引数として渡す)
        segments = transcribe_audio(input_video, device)
        # 2. SRTファイル生成 (deviceを引数として渡す)
        create_srt(segments, temp_srt, device)
        # 3. 字幕焼き込み
        burn_subtitles(input_video, temp_srt, output_video)
        
        print(f"処理が完了しました。出力ファイル: {output_video}")
        
        # GUIモードの場合、完了メッセージを表示
        if args.gui:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("完了", f"処理が完了しました。\n出力ファイル: {output_video}")
        
    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        print(error_msg)
        if args.gui:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("エラー", error_msg)
    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_srt):
            os.remove(temp_srt)

if __name__ == "__main__":
    main() 