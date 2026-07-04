import pandas as pd
import numpy as np
import pywt
from scipy.signal import find_peaks
from scipy.stats import entropy


# DWT FUNCTION (Noise Removal)
def denoise_signal(signal):
    signal = signal.copy()
    signal = np.clip(signal, -1.5, 1.5)
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    threshold = 3 * (np.median(np.abs(coeffs[-1])) / 0.6745)
    new_coeffs = []
    for c in coeffs:
        new_coeffs.append(
            pywt.threshold(c, threshold, mode='soft')
            )
    clean_signal = pywt.waverec(new_coeffs, 'db4')
    clean_signal = clean_signal[:len(signal)]
    return clean_signal

# ECG FEATURES
def detect_r_peaks(signal, fs):
    peaks, _ = find_peaks(
        signal,
        distance=fs * 0.6,
        height=np.mean(signal)
    )
    return peaks


def HR_base(peaks, signal, fs):
    duration = len(signal) / fs
    return (len(peaks) / duration) * 60


def RR_std(peaks, fs):
    rr = np.diff(peaks) / fs
    if len(rr) > 0:
        return np.std(rr)
    return 0


def HR_diff(peaks, fs):

    rr = np.diff(peaks) / fs
    if len(rr) > 0:
        hr = 60 / rr
        return np.std(hr)
    return 0


def qrs_peak(signal):
    return np.max(signal)


def SQI_ecg(signal):
    return np.mean(signal**2) / (np.std(signal) + 1e-6)


# ABP FEATURES
def abp_amp(signal):
    return np.max(signal) - np.min(signal)


def abp_var(signal):
    return np.var(signal)


# PPG FEATURES

def ppg_rate(peaks, signal, fs):
    duration = len(signal) / fs
    return (len(peaks) / duration) * 60


def ppg_std(signal):
    return np.std(signal)


# COMMON FEATURES

def energy(signal):
    return np.sum(signal**2)


def entropy_feature(signal):
    # Convert invalid values
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Remove huge values
    signal = np.clip(signal, -1000, 1000)

    # If signal is constant
    if np.max(signal) == np.min(signal):
        return 0

    try:
        hist, _ = np.histogram(signal, bins=10)
        return entropy(hist + 1e-6)
    
    except:
        return 0


# LOAD DATA
print("Loading Files")

ecg_data = pd.read_csv("all_lead_II.csv")
abp_data = pd.read_csv("all_abp.csv")
ppg_data = pd.read_csv("all_ppg.csv")
labels_data = pd.read_csv("alarm lable.csv")

print("Files Loaded")

# FIXED SAMPLING FREQUENCY
fs = 125


# CREATE DATASET
X = []

patient_columns = ecg_data.columns[1:]
print("Total patients =", len(patient_columns))

# LOOP TO EXRACT SIGNAL OF PATIENTS
print("Feature Extraction Started.")
for patient_col in patient_columns:

    # EXTRACT SIGNALS
    ecg_signal = ecg_data[patient_col].to_numpy().astype(float)
    abp_signal = abp_data[patient_col].to_numpy().astype(float)
    ppg_signal = ppg_data[patient_col].to_numpy().astype(float)

    ecg_signal = np.nan_to_num(ecg_signal)
    abp_signal = np.nan_to_num(abp_signal)
    ppg_signal = np.nan_to_num(ppg_signal)

    # TAKE MIDDLE 12 SECONDS
    num_samples = int(fs * 12)
    total_len = len(ecg_signal)
    mid = total_len // 2

    start = max(0, mid - num_samples // 2)
    end = min(total_len, mid + num_samples // 2)

    ecg_signal = ecg_signal[start:end]
    abp_signal = abp_signal[start:end]
    ppg_signal = ppg_signal[start:end]

    # CHECK MISSING SIGNALS
    abp_missing = np.all(abp_signal == 0)
    ppg_missing = np.all(ppg_signal == 0)

    # DENOISE SIGNALS
    clean_ecg = denoise_signal(ecg_signal)

    if abp_missing:
        clean_abp = abp_signal
    else:
        clean_abp = denoise_signal(abp_signal)

    if ppg_missing:
        clean_ppg = ppg_signal
    else:
        clean_ppg = denoise_signal(ppg_signal)

    # DETECT PEAKS
    ecg_peaks = detect_r_peaks(clean_ecg, fs)

    if ppg_missing:
        ppg_peaks = []
    else:
        ppg_peaks, _ = find_peaks(
            clean_ppg,
            distance=fs * 0.5,
            height=np.mean(clean_ppg)
        )

    # ABP FEATURES
    if abp_missing:
        abp_features = [0, 0]
    else:
        abp_features = [
            abp_amp(clean_abp),
            abp_var(clean_abp)
        ]

    # PPG FEATURES
    if ppg_missing:
        ppg_features = [0, 0]
    else:
        ppg_features = [
            ppg_rate(ppg_peaks, clean_ppg, fs),
            ppg_std(clean_ppg)
        ]

    
    # FINAL FEATURES
    features = [

        # ECG
        HR_base(ecg_peaks, clean_ecg, fs),
        RR_std(ecg_peaks, fs),
        qrs_peak(clean_ecg),
        HR_diff(ecg_peaks, fs),
        SQI_ecg(clean_ecg),

        # ABP
        abp_features[0],
        abp_features[1],

        # PPG
        ppg_features[0],
        ppg_features[1],

        # Common
        energy(clean_ecg),
        entropy_feature(clean_ecg)
    ]

    # Store features
    X.append(features)
print("\nFeature Extraction Completed")

# FINAL DATASET FOR ML
X = np.array(X)
print("\nDataset Shape =", X.shape)


# GET LABELS

y = labels_data["Lable"].to_numpy()
print("Labels Shape =", y.shape)

# MACHINE LEARNING
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining samples =", len(X_train))
print("Testing samples =", len(X_test))
# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining samples =", len(X_train))
print("Testing samples =", len(X_test))

# TRAINING DATA COUNTS
train_false = np.sum(y_train == 0)
train_true = np.sum(y_train == 1)

print("\nTraining False Alarms =", train_false)
print("Training True Alarms =", train_true)

# TESTING DATA COUNTS
test_false = np.sum(y_test == 0)
test_true = np.sum(y_test == 1)

print("\nTesting False Alarms =", test_false)
print("Testing True Alarms =", test_true)
# CREATE MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
print("\nTraining model...")

# TRAIN MODEL
model.fit(X_train, y_train)
print("Model training completed")

# PREDICT
y_pred = model.predict(X_test)
# COUNT FALSE AND TRUE ALARMS

false_alarms = np.sum(y_pred == 0)
true_alarms = np.sum(y_pred == 1)

print("\nTesting False Alarms Detected =", false_alarms)
print("Testing True Alarms Detected =", true_alarms)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy =", accuracy * 100, "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))