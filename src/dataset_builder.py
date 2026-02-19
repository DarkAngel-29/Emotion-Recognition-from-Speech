import os
from extract_mfcc import extract_mfcc
from labels import EMOTION_TO_INDEX
from labels import EMOTION_MAP
import numpy as np

X = []
y = []

dataset_path = "data/audio"

print("Using dataset path:", dataset_path)

print("Dataset path exists:", os.path.exists(dataset_path))

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)

            # Extract label from filename
            emotion_code = file.split("-")[2]
            label = EMOTION_MAP[emotion_code]

            # Extract features
            mfcc = extract_mfcc(path)
            MAX_LEN = 220

            if mfcc.shape[0] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:MAX_LEN, :]
            #print("MFCC shape:", mfcc.shape)
            #print(f"Processed {file}: {label}")
            np.array(X)
            X.append(mfcc)
            y.append(EMOTION_TO_INDEX[label])

X = np.array(X)
y = np.array(y)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Unique labels:", np.unique(y))

np.save("data/features/X.npy", X)
np.save("data/features/y.npy", y)
