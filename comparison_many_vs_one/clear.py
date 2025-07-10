import os
from collections import defaultdict

# 삭제할 디렉토리 지정
target_dir = './comparison_many_vs_one'

# 파일 그룹 딕셔너리
file_groups = defaultdict(list)

# 파일 분류
for fname in os.listdir(target_dir):
    if 'everyday' in fname:
        key = fname.split('everyday', 1)[1]  # 'everyday' 이후의 문자열
        file_groups[key].append(fname)

# 짝수 개의 그룹 삭제
for key, files in file_groups.items():
    if len(files) % 2 == 0:
        print(f"Deleting files with key '{key}':")
        for fname in files:
            full_path = os.path.join(target_dir, fname)
            try:
                os.remove(full_path)
                print(f"  Deleted: {fname}")
            except Exception as e:
                print(f"  Failed to delete {fname}: {e}")
