import cv2
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from scipy import stats

# Initialize the face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def detect_face():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return True
        else:
            return False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

face_detected = detect_face()

if face_detected:
    print("A face was detected!")
else:
    print("No face was detected.")

if face_detected:
    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    webcam = cv2.VideoCapture(0)
    realWidth = 640
    realHeight = 480
    videoWidth = 320
    videoHeight = 240
    videoChannels = 3
    videoFrameRate = 15
    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (20, 30)
    bpmTextLocation = (videoWidth // 2 + 5, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    bpmCalculationFrequency = 15
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    bpm_data = []
    time_data = []
    start_time = time.time()
    bpm_calculation_time = 20
    graph_started = False
    average_bpm = None

    i = 0
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        detectionFrame = frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :]

      
        roi_rgb = detectionFrame[:, :, :]
        avg_rgb = np.mean(roi_rgb, axis=(0, 1))
        print(f"Frame {i}: RGB = {avg_rgb}")

        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)
        fourierTransform[mask == False] = 0

        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
            
            print(f"Frame {i}: Frequency = {hz} Hz")
            
            print(f"Frame {i}: BPM = {bpmBuffer.mean()}")

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize

        frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :] = outputFrame
        cv2.rectangle(frame, (videoWidth // 2, videoHeight // 2), (realWidth - videoWidth // 2, realHeight - videoHeight // 2), boxColor, boxWeight)

        if i > bpmBufferSize:
            cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
        else:
            cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

        if len(sys.argv) != 2:
            cv2.imshow("Webcam Heart Rate Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        current_time = time.time() - start_time
        if len(bpm_data) == 0 or bpmBuffer.mean() != bpm_data[-1]:
            bpm_data.append(bpmBuffer.mean())
        time_data.append(current_time)

        if not graph_started and current_time >= 5:
            graph_started = True
            time_data = [t - 5 for t in time_data]
            start_time = time.time()

        if graph_started and current_time >= 20 and bpm >= 40:
            average_bpm = np.mean(bpm_data)
            break

    webcam.release()
    cv2.destroyAllWindows()

    with open("bpm_values.txt", "w") as file:
        for bpm_value in bpm_data:
            file.write(str(bpm_value) + "\n")

    print("All BPM values:")
    for index, bpm_value in enumerate(bpm_data, 1):
        print(f"Frame {index}: {bpm_value}")

    import matplotlib.pyplot as plt
    from scipy import stats

    def calculate_hrv_stress_index(bpm_data):
        rr_intervals = [60 * 1000 / bpm for bpm in bpm_data]
        time_points = np.cumsum(rr_intervals)
        sdnn = np.std(rr_intervals)
        stress_index = 100 - (sdnn / 10)
        average_hrv = np.mean(rr_intervals)

        with open("hrv_values.txt", "w") as file:
            for hrv_value in rr_intervals:
                file.write(str(hrv_value) + "\n")

        return time_points, rr_intervals, sdnn, stress_index, average_hrv

    def read_bpm_data_from_file(file_path):
        bpm_data = []
        with open(file_path, "r") as file:
            for line in file:
                bpm = float(line.strip())
                bpm_data.append(bpm)
        return bpm_data

    bpm_data = read_bpm_data_from_file("bpm_values.txt")

    time_points, rr_intervals, sdnn, stress_index, average_hrv = calculate_hrv_stress_index(bpm_data)

    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, rr_intervals)
    regression_line = slope * np.array(time_points) + intercept

    time_data = list(range(len(bpm_data)))
    time_points_hrv = [sum(rr_intervals[:i+1]) for i in range(len(rr_intervals))]

    plt.figure(figsize=(10, 9))

    plt.subplot(221)
    plt.plot(time_data[12:], bpm_data[12:], marker='o', label='BPM', linestyle='-', color='blue')
    average_bpm = round(average_bpm, 2)
    plt.title("Heart Rate Graph and Avg BPM :" f' {average_bpm}', transform=plt.gca().transAxes)
    plt.xlabel("Time (Data Point)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.savefig("static/final_graph1.png")

    plt.subplot(222)
    average_hrv = average_hrv / 10
    plt.plot(time_points_hrv[12:], rr_intervals[12:], label='RR Interval')
    average_hrv = round(average_hrv, 2)
    plt.title("HRV Graph and Average HRV :" f' {average_hrv}', transform=plt.gca().transAxes)
    plt.ylim(500, 900)
    plt.xlabel("Time")
    plt.ylabel("RR Interval")
    plt.legend()
    plt.grid(True)
    plt.savefig("static/final_graph2.png")

    plt.subplot(223)
    stress_score = [(hrv - 0.75) * 50 + (bpm - 75) * 0.1 for hrv, bpm in zip(rr_intervals, bpm_data)]
    plt.plot(stress_score[12:], marker='o', linestyle='-', color='red')
    plt.ylim(10000, 75000)
    s = np.mean(stress_score)
    s = s / 1000
    s = round(s, 2)
    plt.title("Stress Level Graph" f' {s}', transform=plt.gca().transAxes)
    plt.xlabel('Time (samples)')
    plt.ylabel('Stress Level')
    plt.grid(True)
    plt.savefig("static/final_graph3.png")

    plt.subplot(224)
    spo2_conversion_factor = 0.1
    spo2_data = [spo2_conversion_factor / hrv for hrv in rr_intervals]
    plt.plot(spo2_data[10:], marker='o', linestyle='-', color='r')
    plt.xlabel('Time (samples)')
    plt.ylabel('SpO2')
    average_spo2 = np.mean(spo2_data)
    average_spo2 = average_spo2 * 1000 * 8
    plt.title(" SpO2 Graph" f' {average_spo2:.2f}', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig("static/final_graph4.png")

    plt.tight_layout()
    plt.show()

    print("Final Readings:")
    print("Average BPM from 5 to 20 seconds:", average_bpm)
    average_hrv = average_hrv / 10
    print("Average HRV (SDNN) (ms):", average_hrv)
    s = np.mean(stress_score)
    s = s / 1000
    print("Average Stress Index:", s)
    average_spo2 = np.mean(spo2_data)
    average_spo2 = average_spo2 * 100000
    print("Average Stress Index:", average_spo2)
