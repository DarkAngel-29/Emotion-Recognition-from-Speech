## 🧭 Project Setup

This project uses an external audio dataset which is also included in the repository.

### 🔽 Step 1 — Download the dataset

If you have issues with downloading the dataset, manually download the dataset zip file from the link below:

[Download Dataset](https://drive.google.com/file/d/1MgV-9f28BLowJ_mPzM7pTSlOGAAKvxTb)

Unzip the downloaded file. You should see folders like:

Actor_01, Actor_02, ..., Actor_24

Move all `Actor_XX` folders into the following directory inside the project:

data/audio/

So your final structure should look like:

```
📂Project/
├── 📂data/
|    ├── 📂audio/
|    |    ├── 🎵Actor_01
|    |    ├── 🎵Actor_02
|    |    ├── ...
|    |    └── 🎵Actor_24
|    └── 📂features/
|        ├── 🤖sample_mfcc.npy
|        ├── 🤖X.npy
|        └── 🤖Y.npy
├── 📂img/
|    └── 📈confusion_matrix_svm.png
├── 📂models/
|    └── ⚙️svm_model.pkl
└── 📂src/
     ├── 📜dataset_builder.py
     ├── 📜extract_mfcc.py
     ├── 📜labels.py.py
     ├── 📜test_audio.py 
     └── 📜train_svm.py   


```
### 🔽 Step 2 — Create a virtual environment (recommended) and install requirements
Create python env
```bash
python -m venv venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
#  or in Terminal
venv\Scripts\activate
pip install -r requirements.txt
```

### 🔽 Step 3 - Run py filea and MODEL

First build dataset

```bash
python src/dataset_builder.py
```

Second run and store model

```bash
python src/train_svm.py
```

### ✅ Final check

Make sure this path exists:

data/audio/Actor_01/

If yes — you're ready to continue with the project 🚀


## 📊 Confution matrix
![Confusion Matrix](img\confusion_matrix_svm.png)