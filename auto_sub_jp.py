import os
import argparse
import whisper
import torch
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import subprocess
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import warnings
from pydantic import BaseModel
from typing import List

class TranslationSegment(BaseModel):
    original: str
    translation: str

class TranslationResponse(BaseModel):
    segments: List[TranslationSegment]

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

def setup_gemini():
    """
    Gemini APIの設定を行う
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEYが設定されていません。.envファイルを確認してください。")
    
    genai.configure(api_key=api_key)
    
    # モデルの設定
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return model

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

def translate_text(segments, device=None):
    """
    英語のテキストを日本語に翻訳
    
    Args:
        segments (list): 翻訳するセグメントのリスト
        device: 未使用（互換性のために残す）
    
    Returns:
        list: 翻訳されたセグメントのリスト
    """
    try:
        model = setup_gemini()
        
        # セグメントを区切りマーク付きで結合
        combined_text = ""
        for i, segment in enumerate(segments):
            combined_text += f"[SEG{i}]{segment['text']}"
        
        # プロンプトの作成
        prompt = f"""
以下の英語テキストを日本語に翻訳してください。テキストは[SEG数字]で区切られたセグメントに分かれています。
文脈を考慮して自然な日本語に翻訳してください。

テキスト:
{combined_text}

[SEG数字]マーカーを使って分割されたテキストを個別のセグメントとして扱い、翻訳してください。
"""
        
        # 翻訳の実行
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "response_mime_type": "application/json",
                "response_schema": TranslationResponse
            }
        )
        
        # レスポンスのテキストを取得して整形
        response_text = response.text.strip()
        
        # デバッグ用の出力
        print("API Response:", response_text)
        
        try:
            # Pydanticモデルとしてパース
            result = TranslationResponse.model_validate_json(response_text)
            
            # 結果をセグメントに反映
            for i, segment in enumerate(segments):
                if i < len(result.segments):
                    segment["translation"] = result.segments[i].translation
                else:
                    print(f"警告: セグメント {i} の翻訳が見つかりません")
                    segment["translation"] = segment["text"]
            
            return segments
            
        except Exception as e:
            print(f"パースエラー: {str(e)}")
            print("受信したテキスト:", response_text)
            raise
        
    except Exception as e:
        print(f"翻訳エラー: {str(e)}")
        # エラーの場合は原文をそのまま返す
        for segment in segments:
            segment["translation"] = segment["text"]
        return segments

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
    
    # プログレスバーの設定
    progress = tqdm(total=2, desc="字幕生成", unit="ステップ", leave=True, dynamic_ncols=True, position=0)
    
    # 1. 全セグメントをまとめて翻訳
    print("翻訳を実行中...")
    translated_segments = translate_text(segments, device)
    progress.update(1)
    
    # 2. SRTファイルの生成
    print("SRTファイルを生成中...")
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(translated_segments, 1):
            # タイムスタンプをフォーマット
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            
            # SRTフォーマットで書き込み
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['translation']}\n\n")
    
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