import os
import glob
import numpy as np

def load_thresholds(threshold_file):
    thresholds = {}
    if not os.path.exists(threshold_file):
        print(f"❌ 에러: {threshold_file} 파일을 찾을 수 없습니다.")
        return None

    with open(threshold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('P') and '|' in line:
                parts = line.split('|')
                try:
                    p_idx = int(parts[0].strip()[1:]) 
                    v2 = float(parts[-2].strip())     
                    v1 = float(parts[-1].strip())     
                    thresholds[p_idx] = {'v2': v2, 'v1': v1}
                except (ValueError, IndexError):
                    continue
    return thresholds

def update_visibility(label_dir, threshold_file, point_count=13):
    thresholds = load_thresholds(threshold_file)
    if not thresholds: return

    detail_files = glob.glob(os.path.join(label_dir, "*_detail.txt"))
    print(f"🔄 총 {len(detail_files)}개의 파일을 분석 및 필터링합니다...")

    changed_files = 0
    total_updated_points = 0
    out_of_bbox_count = 0

    for d_file in detail_files:
        l_file = d_file.replace("_detail.txt", ".txt")
        if not os.path.exists(l_file): continue

        try:
            d_data = np.loadtxt(d_file).reshape(-1, 5 + point_count * 3)
            l_data = np.loadtxt(l_file).reshape(-1, 5 + point_count * 3)
        except Exception as e:
            print(f"⚠️ 로드 실패 ({os.path.basename(d_file)}): {e}")
            continue

        file_modified = False
        
        for row_idx in range(len(d_data)):
            # 1. Bbox 영역 계산 (Normalized coordinates)
            bx, by, bw, bh = d_data[row_idx, 1:5]
            x_min, x_max = bx - bw/2, bx + bw/2
            y_min, y_max = by - bh/2, by + bh/2

            for p_idx in range(point_count):
                # 키포인트 좌표 및 신뢰도 추출
                px, py, conf = d_data[row_idx, 5 + p_idx * 3 : 8 + p_idx * 3]
                current_vis = l_data[row_idx, 5 + p_idx * 3 + 2]

                # --- [추가] Bbox 이탈 여부 체크 ---
                is_out = (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max)
                
                if is_out:
                    new_vis = 0.0
                    new_x, new_y = 0.0, 0.0
                    out_of_bbox_count += 1
                else:
                    # --- 기존 임계값 기반 Visibility 결정 ---
                    new_x, new_y = px, py
                    thres = thresholds.get(p_idx, {'v2': 0.5, 'v1': 0.3})
                    if conf >= thres['v2']:
                        new_vis = 2.0
                    elif conf >= thres['v1']:
                        new_vis = 1.0
                    else:
                        new_vis = 0.0

                # 좌표나 Visibility가 변경되었는지 확인
                if (current_vis != new_vis) or (l_data[row_idx, 5+p_idx*3] != new_x):
                    l_data[row_idx, 5 + p_idx * 3 : 8 + p_idx * 3] = [new_x, new_y, new_vis]
                    file_modified = True
                    total_updated_points += 1

        if file_modified:
            fmt = ['%d'] + ['%.6f'] * 4 + (['%.6f', '%.6f', '%d'] * point_count)
            np.savetxt(l_file, l_data, fmt=fmt)
            changed_files += 1

    print("-" * 60)
    print(f"✅ 작업 완료!")
    print(f" - 수정된 파일 수: {changed_files}개")
    print(f" - 수정된 총 포인트 수: {total_updated_points}개")
    print(f" - Bbox 이탈로 제거된 포인트: {out_of_bbox_count}개")

if __name__ == "__main__":
    label_path = "/home/blake/ai/gemini/yolov7-pose-person/runs/detect/yolov7-pose-person-point-2-/labels"
    threshold_path = "./output/conf_threshold.txt"
    
    update_visibility(label_path, threshold_path, point_count=13)


