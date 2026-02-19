import os
import glob
import numpy as np
import pandas as pd

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1_x1, b1_x2 = x1 - w1/2, x1 + w1/2
    b1_y1, b1_y2 = y1 - h1/2, y1 + h1/2
    b2_x1, b2_x2 = x2 - w2/2, x2 + w2/2
    b2_y1, b2_y2 = y2 - h2/2, y2 + h2/2
    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / (union_area + 1e-9)

def analyze_kpt_confidence(gt_path, pred_path, point_count):
    sigmas = np.array([0.35]*10 + [0.25]*1 + [0.25]*4 + [0.20]*2)
    results = []

    pred_files = glob.glob(os.path.join(pred_path, "*_detail.txt"))
    print(f"🚀 총 {len(pred_files)}개의 파일을 분석 중...")

    for p_file in pred_files:
        base_name = os.path.basename(p_file).replace("_detail.txt", "")
        g_file = os.path.join(gt_path, base_name + ".txt")
        if not os.path.exists(g_file): continue

        p_data = np.loadtxt(p_file).reshape(-1, 5 + point_count * 3)
        g_data = np.loadtxt(g_file).reshape(-1, 5 + point_count * 3)

        for p_row in p_data:
            p_box = p_row[1:5]
            best_iou, best_gt_row = 0, None
            for g_row in g_data:
                iou = calculate_iou(p_box, g_row[1:5])
                if iou > best_iou: best_iou, best_gt_row = iou, g_row

            if best_iou > 0.5:
                scale = p_row[3] * p_row[4]
                for i in range(point_count):
                    px, py, pconf = p_row[5 + i*3 : 8 + i*3]
                    gx, gy, gvis = best_gt_row[5 + i*3 : 8 + i*3]
                    if gvis > 0:
                        dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                        oks = np.exp(-(dist**2) / (2 * (scale + 1e-9) * (sigmas[i]**2)))
                        results.append({'idx': i, 'conf': pconf, 'oks': oks})

    df = pd.DataFrame(results)
    final_recommends = []
    
    # 헤더 정의
    header = f"{'Point':<6} | {'Accuracy':<8} | {'T_High':<7} | {'T_Low':<7} | {'Margin':<8} | {'V2(Visible)':<12} | {'V1(Occlude)':<12}"
    divider = "-" * 110

    output_file = "./output/conf_threshold.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 파일 상단 정보
        f.write("=== Keypoint Auto-Labeling Threshold Analysis ===\n")
        f.write(header + "\n")
        f.write(divider + "\n")

        # 터미널 출력
        print("\n" + "="*110)
        print(header)
        print(divider)

        for i in range(point_count):
            p_df = df[df['idx'] == i]
            if len(p_df) == 0: continue

            high_g = p_df[p_df['oks'] > 0.85]
            mid_g = p_df[(p_df['oks'] > 0.5) & (p_df['oks'] <= 0.85)]
            low_g = p_df[p_df['oks'] < 0.5]

            # 기존 통계치 (T_High, T_Low)
            t_high = high_g['conf'].quantile(0.1) if len(high_g) > 0 else 0.5
            t_low = low_g['conf'].quantile(0.95) if len(low_g) > 0 else 0.2
            margin = t_high - t_low
            status = "✅" if margin > 0 else "⚠️"

            # 평균 방식 임계값 계산 (V2, V1)
            v2_m1 = high_g['conf'].quantile(0.15) if len(high_g) > 0 else 0.6
            v2_m2 = max(low_g['conf'].quantile(0.95) if len(low_g) > 0 else 0, high_g['conf'].median() * 0.7 if len(high_g) > 0 else 0.5)
            v2_final = (v2_m1 + v2_m2) / 2

            v1_m1 = mid_g['conf'].quantile(0.10) if len(mid_g) > 0 else 0.3
            v1_m2 = mid_g['conf'].median() * 0.6 if len(mid_g) > 0 else v2_final * 0.5
            v1_final = (v1_m1 + v1_m2) / 2

            accuracy = p_df['oks'].mean()
            
            # 한 줄 데이터 생성
            row_str = f"P{i:02d}     | {accuracy:.4f}   | {t_high:.3f}   | {t_low:.3f}   | {margin:+.3f} {status} | {v2_final:.4f}     | {v1_final:.4f}"
            
            print(row_str)
            f.write(row_str + "\n")
            
            final_recommends.append({'v2': v2_final, 'v1': v1_final})

        # 요약 정보 계산 및 출력
        avg_v2 = np.mean([r['v2'] for r in final_recommends])
        avg_v1 = np.mean([r['v1'] for r in final_recommends])
        
        summary = (
            f"{divider}\n"
            f"📌 [FINAL RECOMMENDATION]\n"
            f" * Global V2 (Visible)  Threshold: {avg_v2:.4f}\n"
            f" * Global V1 (Occluded) Threshold: {avg_v1:.4f}\n"
            f"{divider}"
        )
        
        print(summary)
        f.write(summary + "\n")

    print(f"\n✅ 분석 데이터가 '{output_file}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    analyze_kpt_confidence(
        gt_path = "/home2/ai/data/pose/coco-pose/validation/",
        pred_path = "/home/blake/ai/gemini/yolov7-pose-person/runs/detect/yolov7-pose-person-point-2-3/labels",
        point_count = 13
    )

