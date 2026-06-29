import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import json
import numpy as np
import unicodedata
import re
import uuid
from tqdm import tqdm
from paddleocr import PaddleOCR

# 1. Initialize OCR Engine (Japanese)
# ゲームUIは平坦なデジタルのため、ドキュメント用の傾き補正や湾曲補正を無効化して高速化・座標歪みを防止する
ocr = PaddleOCR(lang='japan', use_doc_orientation_classify=False, use_doc_unwarping=False)

def preprocess_image(roi_img):
    """
    ゲームUI特有のノイズ除去とコントラスト強化の前処理
    """
    # グレースケール化
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 必要に応じてリサイズ（OCR精度向上のため）
    # gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # コントラスト強調などの前処理を追加可能
    # _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # PaddleOCRは3チャンネル画像(BGR/RGB)を想定しているため、
    # 処理後のグレースケール画像を3チャンネルに変換して返す
    processed_img_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return processed_img_3c

def classify_text_color(roi_bgr):
    """
    文字色を判定する（白=merit, 水色系=demerit）
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    bright_pixels = roi_bgr[mask == 255]
    if len(bright_pixels) == 0:
        return "merit"
        
    avg_b = np.mean(bright_pixels[:, 0])
    avg_r = np.mean(bright_pixels[:, 2])
    
    # 水色（デメリット）は青成分(B)が赤成分(R)より大幅に大きい
    if (avg_b - avg_r) > 30:
        return "demerit"
    else:
        return "merit"

def clean_ocr_text(raw_text):
    """
    OCRで認識されたテキストのクレンジングと正規化
    """
    if not raw_text:
        return ""
    
    # NFKC正規化（全角英数字・記号を半角に、半角カタカナを全角にするなど）
    text = unicodedata.normalize('NFKC', raw_text)
    
    # ゲームUIのOCRで発生しやすい典型的な誤認識の置換
    replacements = {
        "力ッ卜": "カット",
        "力ッ": "カッ",
        "ッ卜": "ット",
        "被发": "被ダメ",
        "波ダメージ": "被ダメージ",
        "彼ダメージ": "被ダメージ",
        "発×": "発狂",
        "×一ジ": "ージ",
        "物理攻擊": "物理攻撃",
        "剌突": "刺突",
        "波デ": "被デ",
        "？": "?",
        "：": ":",
        "＋": "+",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text.strip()

def correct_relic_name(name):
    if not name:
        return name
        
    prefixes = ["壮大な", "端正な", "繊細な"]
    middles = ["燃える", "滴る", "輝く", "静まる"]
    suffixes = ["景色", "昏景"]
    
    # 24通りの組み合わせを生成
    combinations = []
    for p in prefixes:
        for m in middles:
            for s in suffixes:
                combinations.append(p + m + s)
                
    # 固有アイテムのリスト (wiki参照)
    unique_relics = [
        "ちぎれた組み紐", "にび色の砥石", "ガラスの首飾り", "三冊目の本", "割れた封蝋",
        "古びたミニアチュール", "古びた懐中時計", "夜の痕跡", "安寧の遺志", "安寧者の遺志",
        "小さな化粧道具", "忌み鬼の呪物", "深海の夜", "深海の暗き夜", "清浄の雫",
        "爵の夜", "爵の暗き夜", "片眼鏡の革袋", "狩人の夜", "狩人の暗き夜", "獣の夜",
        "獣の暗き夜", "王の夜", "瓦礫の夜", "石の杭", "祝福された花",
        "祝福された鉄貨", "聖律の刃", "薄汚れたフレーム", "記録「後継者へ」", "識の夜",
        "識の暗き夜", "追跡者の耳飾り", "金色の露", "銀の雫", "霞の夜", "霞の暗き夜",
        "頭冠のメダル", "骨のような石", "魔の夜", "魔の暗き夜", "魔女のブローチ",
        "砕けた魔女のブローチ", "黄金の萌芽", "黒爪の首飾り"
    ]
    
    # 重複排除してマージ
    all_candidates = list(set(combinations + unique_relics))
                
    # Levenshtein距離を計算する関数
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    best_match = None
    min_dist = 999
    
    for combo in all_candidates:
        dist = levenshtein(name, combo)
        if dist < min_dist:
            # 許容する最大誤認識数を文字数に応じて動的に決める (文字数の約3分の1、最低1)
            allowed_error = max(1, len(combo) // 3)
            if dist <= allowed_error:
                min_dist = dist
                best_match = combo
                
    if best_match is not None:
        return best_match
        
    return name

def process_batch(batch_frames, extracted_data_list, ocr):
    if not batch_frames:
        return

    batch_texts = [{"title": [], "effects": []} for _ in range(len(batch_frames))]

    for frame_idx, (roi_img, _) in enumerate(batch_frames):
        # OCR処理を高速化するため、画像を0.7倍に縮小（精度を落とさずに約25%高速化）
        h_roi, w_roi = roi_img.shape[:2]
        scale = 0.7
        resized_roi = cv2.resize(roi_img, (int(w_roi * scale), int(h_roi * scale)), interpolation=cv2.INTER_AREA)

        # ROIをテンポラリPNGファイルに保存してOCRを実行（NumPy直渡しによるoneDNNエラー回避）
        tmp_path = f"_tmp_roi_{frame_idx}.png"
        try:
            cv2.imwrite(tmp_path, resized_roi)
            results = ocr.predict(tmp_path)
            if not results:
                continue
            res = results[0]  # 1ファイル = 1結果

            roi_h, roi_w = roi_img.shape[:2]

            texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
            polys = res.get('dt_polys', []) if isinstance(res, dict) else getattr(res, 'dt_polys', [])

            for i, text in enumerate(texts):
                if not text or not isinstance(text, str):
                    continue
                text = clean_ocr_text(text)
                if not text:
                    continue
                try:
                    poly = polys[i]
                    # 縮小画像での座標を元の画像サイズにスケールバック
                    y_coords = [p[1] / scale for p in poly]
                    x_coords = [p[0] / scale for p in poly]
                    y_center = sum(y_coords) / len(y_coords)

                    # Color classification (Merit vs Demerit)
                    crop_y1 = int(max(0, min(y_coords)))
                    crop_y2 = int(min(roi_h, max(y_coords)))
                    crop_x1 = int(max(0, min(x_coords)))
                    crop_x2 = int(min(roi_w, max(x_coords)))

                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        line_crop = roi_img[crop_y1:crop_y2, crop_x1:crop_x2]
                        color_type = classify_text_color(line_crop)
                    else:
                        color_type = "merit"

                    line_data = {"type": color_type, "text": text, "y": y_center}

                    # Y_centerを基準にタイトルとエフェクトを分離
                    if y_center < roi_h * (46.5 / 225):
                        batch_texts[frame_idx]["title"].append(line_data)
                    else:
                        batch_texts[frame_idx]["effects"].append(line_data)
                except (IndexError, TypeError, KeyError):
                    continue
        except Exception as e:
            print(f"OCR error (frame {frame_idx}): {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ソートとJSON変換
    for idx, texts_by_region in enumerate(batch_texts):
        # 画像の形式に合わせたフラットな構造を定義
        flat_item = {
            "relic_name": "",
            "skill1": "",
            "skill1_demerit": "",
            "skill2": "",
            "skill2_demerit": "",
            "skill3": "",
            "skill3_demerit": "",
            "item_id": str(uuid.uuid4()),
            "tags": [],
            "note": ""
        }
        
        # タイトルの処理
        title_lines = texts_by_region.get("title", [])
        if title_lines:
            title_lines.sort(key=lambda x: x["y"])
            raw_title = " ".join([L["text"] for L in title_lines]).strip()
            flat_item["relic_name"] = correct_relic_name(raw_title)
            
        # エフェクトの処理
        effect_lines = texts_by_region.get("effects", [])
        effect_blocks = []
        if effect_lines:
            effect_lines.sort(key=lambda x: x["y"])
            
            roi_h = batch_frames[idx][0].shape[0]
            current_block = []
            
            for L in effect_lines:
                if not current_block:
                    # ブロックの先頭は必ずメリット（仕様上デメリット先頭はないが念のため）
                    current_block.append(L)
                else:
                    gap = L["y"] - current_block[-1]["y"]
                    
                    if L["type"] == "demerit":
                        # デメリットは常に現在のブロックに追加（先頭にはならない）
                        current_block.append(L)
                    else:
                        # メリットの場合：ブロックが2行以上、またはギャップが大きければ新ブロック
                        if len(current_block) >= 2 or gap > roi_h * 0.13:
                            effect_blocks.append(current_block)
                            current_block = [L]
                        else:
                            current_block.append(L)
                            
            if current_block:
                effect_blocks.append(current_block)
                
            # 各効果ブロックを平坦な構造にマッピング（複数行の同タイプは自動マージ）
            for i, block in enumerate(effect_blocks[:3]):
                skill_idx = i + 1
                merits = []
                demerits = []
                for L in block:
                    cleaned_val = L["text"].strip()
                    # 1. 先頭の1文字のゴミ（アイコンの誤認識等）＋スペースを除去
                    while len(cleaned_val) > 2 and cleaned_val[1] == " ":
                        cleaned_val = cleaned_val[2:].strip()
                    # 2. 2文字以下の短い行はOCRノイズとみなしてスキップ
                    if len(cleaned_val) <= 2:
                        continue
                        
                    if L["type"] == "merit":
                        merits.append(cleaned_val)
                    else:
                        demerits.append(cleaned_val)
                
                if merits:
                    flat_item[f"skill{skill_idx}"] = " ".join(merits)
                if demerits:
                    flat_item[f"skill{skill_idx}_demerit"] = " ".join(demerits)
        
        # 遺物名か効果のいずれかが抽出されている場合のみ保存
        if flat_item["relic_name"] or flat_item["skill1"] or flat_item["skill2"] or flat_item["skill3"]:
            extracted_data_list.append(flat_item)

def main():
    video_path = 'game_play_record.mp4'
    
    # 動画ファイルが存在するかチェック
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Please place the video file in the project directory.")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted_data_list = []
    frame_count = 0
    batch_frames = [] # バッチ処理用のバッファ
    
    # 変化検知用の変数
    prev_roi_gray = None # 最後にOCRを実行した確定フレーム
    current_slot_frames = [] # 現在のスロット内の全フレーム
    
    # グレースケール平均差分の閾値（0〜255のピクセル値の平均差）
    # 小さくするほど敏感になる（4〜5フレームで切り替わるアイテムに対応）
    # 0.5に設定することで、テキストの類似度が高いアイテム間の切り替わりを漏れなく検知します
    mean_diff_threshold = 0.5
    
    print("動画解析を開始します。画面の切り替わりを自動検知してOCRを実行します...")
    
    # tqdmでプログレスバーを表示
    pbar = tqdm(total=total_frames, desc="動画を解析中", unit="frame")
    
    try:
        # 処理ループ
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. ROI (Region of Interest) extraction
            h, w = frame.shape[:2]
            roi_y1 = int(h * 0.7083)
            roi_y2 = int(h * 0.9093)
            roi_x1 = int(w * 0.5651)
            roi_x2 = int(w * 0.9411)
            
            # 安全のためクリッピング
            roi_y1, roi_y2 = max(0, roi_y1), min(h, roi_y2)
            roi_x1, roi_x2 = max(0, roi_x1), min(w, roi_x2)
            
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # 前フレームとの差分を比較（RAWグレースケールで比較）
            roi_gray_small = cv2.resize(roi_gray, (64, 32))
            
            is_changed = False
            
            if prev_roi_gray is None:
                is_changed = True
            else:
                diff = cv2.absdiff(roi_gray_small.astype(np.float32),
                                   prev_roi_gray.astype(np.float32))
                mean_diff = float(np.mean(diff))
                
                # 平均輝度差がthresholdを超えたら「テキストが切り替わった」と判定
                if mean_diff > mean_diff_threshold:
                    is_changed = True
            
            # 画面が切り替わった場合、完了したスロットの中央フレームを処理対象にする
            if is_changed:
                if current_slot_frames:
                    # 前のスロットの中央フレームを選択してバッファに追加
                    mid_idx = len(current_slot_frames) // 2
                    mid_roi, mid_frame_idx = current_slot_frames[mid_idx]
                    batch_frames.append((mid_roi.copy(), mid_frame_idx))
                    
                    # バッファが4件溜まったら一括処理
                    if len(batch_frames) >= 4:
                        process_batch(batch_frames, extracted_data_list, ocr)
                        batch_frames = []
                    
                    current_slot_frames = []
                
                prev_roi_gray = roi_gray_small.copy()
            
            # 現在のスロットにフレームを追加
            current_slot_frames.append((roi_frame, frame_count))
            
            frame_count += 1
            pbar.update(1)
            
        # 最後のスロットの中央フレームを処理
        if current_slot_frames:
            mid_idx = len(current_slot_frames) // 2
            mid_roi, mid_frame_idx = current_slot_frames[mid_idx]
            batch_frames.append((mid_roi.copy(), mid_frame_idx))
            
        # 動画終了時にバッファに残っている分を処理
        if len(batch_frames) > 0:
            process_batch(batch_frames, extracted_data_list, ocr)
            batch_frames = []

    except KeyboardInterrupt:
        print("\n[中断] ユーザーによって処理が中断されました。途中までのデータを保存します。")
        
    finally:
        pbar.close()
        cap.release()
    
        # 6. Deduplicate consecutive identical items by text
        deduplicated_list = []
        for item in extracted_data_list:
            if not deduplicated_list:
                deduplicated_list.append(item)
            else:
                last = deduplicated_list[-1]
                is_dup = True
                for k in ['relic_name', 'skill1', 'skill1_demerit', 'skill2', 'skill2_demerit', 'skill3', 'skill3_demerit']:
                    if item.get(k) != last.get(k):
                        is_dup = False
                        break
                if not is_dup:
                    deduplicated_list.append(item)

        output_json = 'game_stats.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_list, f, ensure_ascii=False, indent=4)
            
        print(f"Extraction complete. Raw detections: {len(extracted_data_list)}, Deduplicated items: {len(deduplicated_list)}")
        print(f"Data saved to {output_json}")

if __name__ == "__main__":
    main()
