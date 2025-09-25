import random

input_file = "everyday_val.txt"
output_file = "everyday_val_small.txt"

# 파일 읽기
with open(input_file, "r") as f:
    lines = f.readlines()

# 파트 개수가 002인 줄만 추리기
filtered = [line.strip() for line in lines if line.startswith("002 ")]

# 랜덤으로 50개 뽑기 (개수가 적으면 그냥 전체 사용)
sampled = random.sample(filtered, min(50, len(filtered)))

# 새 파일로 저장
with open(output_file, "w") as f:
    f.write("\n".join(sampled))

print(f"{len(sampled)}개를 {output_file}에 저장했습니다.")
