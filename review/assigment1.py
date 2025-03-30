import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os

filtered_image_dir = "filtered_images"
process_image_dir = "process_images"
valid_peaks_dir = "valid_peaks_plots"
os.makedirs(filtered_image_dir, exist_ok=True)
os.makedirs(process_image_dir, exist_ok=True)
os.makedirs(valid_peaks_dir, exist_ok=True)

chunk_size = 300
overlap = 30
file_data = []

with open("data/assignment_1_raw_data.txt", 'r') as file:
    lines = file.readlines()
    for line in lines[15:-1]:
        values = line.strip().split()
        second_int = int(values[1])
        file_data.append(second_int)

np.savetxt("output/second_array.csv", file_data, delimiter=',')

pk_list = []
x_result = []
valid_peak_shapes = []  # 유효한 피크 파형 저장 리스트
exc = 0  # 예외 처리된 청크 개수

for i in range(0, len(file_data), chunk_size - overlap):
    x_chunk = file_data[i:i + chunk_size]
    filtered = hp.filter_signal(x_chunk, [0.5, 8], sample_rate=25, order=3, filtertype='bandpass')

    try:
        wd, m = hp.process(filtered, sample_rate=25)

        peak_count = max(0, len(wd['peaklist']) - len(wd['removed_beats']))
        pk_list.append(peak_count)

        peaks = wd['peaklist']
        fake_peaks = wd['removed_beats']
        real_peaks = [item for item in peaks if item not in fake_peaks]

        print(f"{i}번째 청크 - 유효한 피크 개수: {len(real_peaks)}")

        if peak_count > 0:
            x_result.append(wd['hr'])

            peak_shapes = []
            for index in real_peaks:
                if not ((index - 13 < 0) or (index + 14 >= len(filtered))):
                    peak_shape = filtered[index - 13:index + 14]
                    peak_shapes.append(peak_shape)

            valid_peak_shapes.append(peak_shapes)

    except Exception as e:
        print(f"예외 발생 ({i}번째 청크): {e}")
        exc += 1
        continue

# 평균 이상 피크를 가진 블록만 저장
if len(pk_list) == 0:
    print("Warning: No valid peak data found. 모든 청크에서 피크가 감지되지 않음.")
    avg_peak_count = 0
else:
    pk_np = np.array(pk_list)
    avg_peak_count = np.mean(pk_np)

print(f"전체 블록 평균 초록색 피크 개수: {avg_peak_count:.2f}")

# 평균 이상 피크를 가진 블록만 필터링
valid_peak_shapes_filtered = [valid_peak_shapes[j] for j in range(len(valid_peak_shapes)) if pk_np[j] > avg_peak_count]

plt.figure(figsize=(10, 4))
for idx, peak_shapes in enumerate(valid_peak_shapes_filtered):
    for peak in peak_shapes:
        plt.plot(peak, alpha=0.6)
plt.title(f"Valid Peaks in Chunk")
plt.xlabel("Time")
plt.ylabel("Signal Amplitude")
plt.savefig(os.path.join(valid_peaks_dir, f"valid_peaks.png"))
plt.close()

print(f"총 {len(pk_np)}개의 블록 중 평균 이상 피크를 가진 {len(valid_peak_shapes_filtered)}개의 블록이 저장됨")