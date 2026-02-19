import os
import shutil
import math
import random
from tqdm import tqdm

def split_files(src_dir, dest_dir, n, classes_txt_path):
    # src_dir: 원본 디렉터리 경로
    # dest_dir: 새로 생성될 디렉터리의 루트 경로
    # n: 파일을 나눌 폴더의 파일 수
    # classes_txt_path: classes.txt 파일 경로

    # 디렉터리 내 모든 .jpg 및 .png 파일 리스트 가져오기
    image_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)
    total_files = len(image_files)
    num_dirs = math.ceil(total_files / n)  # 필요한 디렉터리 수 계산

    for i in tqdm(range(num_dirs), desc="Splitting files into folders"):
        # 폴더 이름 지정
        folder_name = f"split_{n}_{i}"
        target_dir = os.path.join(dest_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)  # 폴더 생성

        # classes.txt 파일 복사
        if os.path.exists(classes_txt_path):
            shutil.copy(classes_txt_path, target_dir)

        # 각 폴더에 들어갈 파일 범위 설정
        start_idx = i * n
        end_idx = min(start_idx + n, total_files)
        files_to_move = image_files[start_idx:end_idx]

        for file in tqdm(files_to_move, desc=f"Processing folder {folder_name}", leave=False):
            # 파일 경로 설정
            img_path = os.path.join(src_dir, file)
            txt_path = os.path.join(src_dir, file.rsplit('.', 1)[0] + '.txt')

            # 이미지 및 txt 파일을 타겟 디렉터리로 이동
            shutil.move(img_path, os.path.join(target_dir, file))
            if os.path.exists(txt_path):
                shutil.move(txt_path, os.path.join(target_dir, os.path.basename(txt_path)))

    print(f"{num_dirs}개의 폴더에 파일이 나누어졌습니다.")

# 사용 예시
# 130_OFFICE_Data_Done: 8000
# 130_OFFICE_Data_Done: 2496
# 130_OFFICE_Data_Done: 1681
# 132_BACKGROUND_Data_done: 1070
# 126_TN_data_Done
# 123_KAGGLE_Data_Done
# classes.txt  driveway_walk_done  falling_done  fight_done  jay_walk_done  merge.py  putup_umbrella_done

src_directory = '/data/HERMES/pose_detector/Annotation/Person/119_IR_Data_Person'  # 원본 디렉터리 경로
destination_directory = '/data/HERMES/pose_detector/Annotation/Person/119_IR_Data_Person' # 대상 디렉터리 경로
classes_txt_path = '' # '/data/ARGOS/face_detector/script/classes.txt' # classes.txt 파일 경로
n_value = 100  # 각 폴더에 넣을 파일 수
split_files(src_directory, destination_directory, n_value, classes_txt_path)

exit
 
#src_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/fight_done'  # 원본 디렉터리 경로
#destination_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/fight_done' # 대상 디렉터리 경로
#split_files(src_directory, destination_directory, n_value, classes_txt_path)

#src_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/jay_walk_done'  # 원본 디렉터리 경로
#destination_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/jay_walk_done' # 대상 디렉터리 경로
#split_files(src_directory, destination_directory, n_value, classes_txt_path)

#src_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/putup_umbrella_done'  # 원본 디렉터리 경로
#destination_directory = '/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/putup_umbrella_done' # 대상 디렉터리 경로
#split_files(src_directory, destination_directory, n_value, classes_txt_path)

