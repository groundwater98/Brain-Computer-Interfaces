import numpy as np
import mne
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, recall_score, precision_score
import tensorflow as tf
from pprint import pprint
from model import build_FNN40_basis
from sklearn.utils import class_weight

def load_raw_trials(subject, session):
    # BNCI2014_001 (BCI Competition IV 2A) 데이터셋을 로드
    dataset = BNCI2014_001()

    # 특정 subject(participant)의 특정 session(training or validation)을 가져옴
    runs = dataset._get_single_subject_data(subject)[session]

    # 데이터를 저장할 리스트
    X_all, y_all = [], []

    # Sampling frequency (처음 run에서 추출 후 고정)
    sfreq = None

    # Session 내의 각 run(파일 단위)마다 반복
    for run_key in runs.keys():
        raws = runs[run_key]

        # 첫 번째 run에서 sampling frequency 저장 : 255
        if sfreq is None:
            sfreq = int(raws.info["sfreq"])
            print(f"  Sampling frequency (sfreq): {sfreq} Hz")

        # 이벤트 추출 (자극 채널 'stim'에서 이벤트 탐지)
        events = mne.find_events(raws, stim_channel='stim', shortest_event=0, verbose=False)

        # 이벤트가 없으면 이 run은 건너뜀
        if len(events) == 0:
            print("    [Warning] No events found in this run.")
            continue

        # Imagined movement class: left(1), right(2), feet(3), tongue(4)
        movement_ids = [1, 2, 3, 4] 
        movement_events = np.array([e for e in events if e[2] in movement_ids])

        # Class label 매핑 (MNE의 Epochs에서 사용)
        event_id = {'left': 1, 'right': 2, 'feet': 3, 'tongue': 4}
        '''
        각 movement event에 대해 trial 추출
        - tmin=0, tmax=8: 자극 시점부터 8초까지의 전체 trial 구간 사용
        - baseline=None: baseline 보정 안 함
        - preload=True: 데이터를 즉시 메모리에 로딩
        '''
        epochs = mne.Epochs(raws, movement_events, event_id=event_id, tmin=0, tmax=8, baseline=None, preload=True)
        
        # 22 channel EEG만 추출 (EOG 등 제외), shape: (n_trials, 22, time_points)
        X = epochs.get_data()[:, :22, :]

        # 해당 trial들의 class label 정보 추출
        y_events = epochs.events[:, 2]  
        
        # 결과 누적
        X_all.append(X)
        y_all.append(y_events)

    # 여러 run에서 수집한 데이터를 하나로 합침
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # EEG 데이터, 레이블, 샘플링 주파수 반환
    return X_all, y_all, sfreq

def get_7class_labels(trial_event, sfreq, trial_len=8.0):
    # trial 하나에 대한 시간 축의 각 시점에 해당하는 label을 저장할 배열 생성
    labels = np.zeros(int(trial_len * sfreq), dtype=int)
    # Class 1: [0s ~ 2s] trial 간 break
    labels[:int(2*sfreq)] = 1
    # Class 3: [2s ~ 2.5s] cue가 나타나기 전 대기 시간
    labels[int(2*sfreq):int(2.5*sfreq)] = 3
    # Class 2: [2.5s ~ 3s] cue와 imagined movement 사이 시간
    labels[int(2.5*sfreq):int(3*sfreq)] = 2

    # Class 4 ~ 7: [3s ~ 6s] left, right, feet, tongue를 각각 4 ~ 7로 mapping
    movement_class = 4 + (trial_event - 1)
    labels[int(3*sfreq):int(6*sfreq)] = movement_class

    # Class 1: [6s ~ 8s] tral 종료 후 short break
    labels[int(6*sfreq):] = 1
    return labels

def preprocess_trials(X):
    X_proc = []
    for trial in X:
        # (1) Smoothing: Savitzky-Golay filter (moving local polynomial regression)
        '''
        논문 3.1.1의 수식 (2)에 기반한 Local Polynomial Estimator 적용

        여기서는 Savitzky-Golay 필터로 이를 실용적으로 구현:
        - window_length=5 : 논문에서의 bandwidth h 에 대응됨
        - polyorder=3 : 논문 수식에서의 다항식 차수 p 에 대응됨
        - axis=-1 : 시간축 기준으로 필터링 수행 (trial shape: [channels, time])
        '''
        trial = savgol_filter(trial, window_length=5, polyorder=3, axis=-1)
        # (2) Standardization: channel-wise Z-score normalization
        '''
        논문 3.1.2 Normalization

        이는 함수형 Z-score 정규화와 동일한 구조로,
        각 채널(함수)을 평균 0, 분산 1로 정규화함.
        
        코드에서는 discrete version으로 다음을 수행:
        - trial.T: shape을 (time, channels)로 바꿔 각 채널을 feature로 봄
        - StandardScaler().fit_transform(): 각 채널에 대해 평균과 표준편차를 구해 Z-score 정규화
        - 다시 .T로 전치하여 원래 (channels, time) 형태로 복원
        '''
        trial = StandardScaler().fit_transform(trial.T).T
        X_proc.append(trial)
    return np.array(X_proc)

def make_sliding_windows(X, labels_7class, window_size=1.0, step_size=0.004, sfreq=250):
    # window 길이와 step size를 sample 수 단위로 변환
    '''
    1s - 250 sample
    0.004s - 1 sample 간격
    '''
    win_len = int(window_size * sfreq)
    step = int(step_size * sfreq)
    windows, y_windows = [], []

    # trial 별로 EEG와 label을 하나씩 꺼내며 반복
    for trial_data, trial_labels in zip(X, labels_7class):
        # trial 전체를 sliding window 방식으로 분할할
        for start in range(0, trial_data.shape[1] - win_len + 1, step):
            # 현재 시작 시점부터 window 길이 만큼 잘라냄, shape: (channels, time)
            window = trial_data[:, start:start+win_len]
            # 모델 입력을 위해 (time, channels)로 transpose
            window = window.T 
            # 해당 window 내의 label 중 가장 많이 등장한 class 선택 (majority voting)
            window_label = np.bincount(trial_labels[start:start+win_len]).argmax()

            # 결과 저장
            windows.append(window)
            y_windows.append(window_label)
    return np.array(windows), np.array(y_windows)

# TensorFlow Dataset 생성 함수
def make_tf_dataset_generator(X, y, batch_size=32):

    def generator():
        # X: [window 개수, time, channel], y: [window 개수]
        for xi, yi in zip(X, y):
            # 각 window와 label을 yield
            yield xi.astype(np.float32), int(yi)

    ds = tf.data.Dataset.from_generator(
        generator,
        # window: [time, channel], label: scalar
        output_signature=(
            tf.TensorSpec(shape=(X.shape[1], X.shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
    )

    # batch 단위로 묶음
    ds = ds.batch(batch_size)

    return ds

# 실험 결과를 저장할 리스트트
results = []
with open("result.txt", "a") as f:
    # Hyperparameter로 어떤 subject에 대해 실험을 진행할지 설정
    subject = 8
    # 실험 대상 번호
    msg_subject = f"\n=== Subject {subject} ==="
    print(msg_subject)
    f.write(msg_subject + "\n")

    # 1. BCI Competition IV 2A 데이터 로드 (train, test 분리)
    X_train, y_events_train, sfreq = load_raw_trials(subject, "0train")
    X_test, y_events_test, _ = load_raw_trials(subject, "1test")

    # 2. 논문의 전처리(smoothing, normalization)
    X_train = preprocess_trials(X_train)
    X_test = preprocess_trials(X_test)

    # 3. 7 class로 변환
    labels_7class_train = [get_7class_labels(ev, sfreq) for ev in y_events_train]
    labels_7class_test = [get_7class_labels(ev, sfreq) for ev in y_events_test]

    # 4. sliding window 분할
    Xw_train, yw_train = make_sliding_windows(X_train, labels_7class_train, sfreq=sfreq)
    Xw_test, yw_test = make_sliding_windows(X_test, labels_7class_test, sfreq=sfreq)

    # 5. test 데이터 window 개수 제한 (메모리 문제로 50000로 설정, 논문에는 정확한 해당 내용이 없다.)
    max_test = 50000
    Xw_test, yw_test = Xw_test[:max_test], yw_test[:max_test]

    # 6. tf.data.Dataset 생성
    train_ds = make_tf_dataset_generator(Xw_train, yw_train, batch_size=32)
    train_ds = train_ds.shuffle(10000).repeat()

    # 7. 10번 반복
    for repeat in range(10):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        msg_repeat = f"  Repeat {repeat+1}/10"
        print(msg_repeat)
        f.write(msg_repeat + "\n")

        # FNN(40) 모델 생성
        model = build_FNN40_basis(input_shape=(250, 22), num_classes=8, q=5)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        model.summary()

        # EPOCH 별 학습 횟수 (전체 train window 개수 // batch size)
        steps_per_epoch = len(Xw_train) // 32
        model.fit(train_ds, epochs=30, steps_per_epoch=steps_per_epoch, verbose=1)

        # prediction 수행
        y_pred_probs = model.predict(Xw_test, batch_size=128, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = yw_test

        # 평가 지표 accuracy, recall, precision 계산
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results.append([subject, repeat, acc, rec, prec])

        # 출력 로그 및 저장
        msg_result = f"    ACC: {acc:.4f} / RECALL: {rec:.4f} / PRECISION: {prec:.4f}"
        print(msg_result)
        f.write(msg_result + "\n")

    # 평균 결과 계산
    results_np = np.array(results)  # shape: (10, 5)
    mean_acc = np.mean(results_np[:, 2])
    mean_rec = np.mean(results_np[:, 3])
    mean_prec = np.mean(results_np[:, 4])

    msg_avg = (
        f"\n  === AVERAGE RESULT FOR SUBJECT {subject} ===\n"
        f"    ACCURACY:  {mean_acc:.4f}\n"
        f"    RECALL:    {mean_rec:.4f}\n"
        f"    PRECISION: {mean_prec:.4f}"
    )
    print(msg_avg)
    f.write(msg_avg + "\n")