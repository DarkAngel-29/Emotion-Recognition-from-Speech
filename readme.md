## 📁 Dataset Setup

This project uses an external audio dataset which is not included in the repository.

### 🔽 Step 1 — Download the dataset

Download the dataset zip file from the link below:

[Download Dataset](https://drive.google.com/file/d/1MgV-9f28BLowJ_mPzM7pTSlOGAAKvxTb)

### 📂 Step 2 — Extract the dataset

Unzip the downloaded file. You should see folders like:

Actor_01, Actor_02, ..., Actor_24

### 📁 Step 3 — Place the folders in the correct location

Move all `Actor_XX` folders into the following directory inside the project:

data/audio/

So your final structure should look like:

```
Project/
├── data/
|    ├── audio/
|    |    ├── Actor_01
|    |    ├── Actor_02
|    |    ├── ...
|    |    └── Actor_24
|    └── features/
|        ├── sample_mfcc.npy
|        ├── X.npy
|        └── Y.npy
```


### ✅ Final check

Make sure this path exists:

data/audio/Actor_01/

If yes — you're ready to continue with the project 🚀
