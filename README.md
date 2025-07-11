# 🐱🐶 Cat and Dog Image Classifier

This Python project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images as either **"Cat"** or **"Dog"**. It supports model training, validation, testing on new images, data augmentation, and performance logging with TensorBoard.

---

## 🚀 Features

- 📚 **Custom CNN (CatDogNet)**: Designed with multiple convolutional and fully connected layers.
- 🧪 **Data Augmentation**: Uses `albumentations` for advanced image transformations (crop, flip, rotate, brightness/contrast).
- 📈 **Training & Validation**: Full training loop with accuracy/loss metrics printed and logged.
- 🧠 **Inference**: Predict the class of new images with confidence scoring.
- 💾 **Model Saving**: Saves the trained model as `cat_dog_model.pth`.
- 📊 **TensorBoard Integration**: Training metrics are logged for visualization.

---

## 🧰 Prerequisites

Install dependencies:

```bash
pip install torch torchvision numpy Pillow albumentations tensorboard
```

---

## 📦 Dataset

Expected directory structure (from Kaggle's [Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)):

```
PetImages/
├── Cat/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── Dog/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

Place the `PetImages/` folder in the same directory as `main.py`.

---

## 🏁 How to Use

### 1. Train the Model

Uncomment the training line in `main.py`:

```python
if __name__ == "__main__":
    train_model("PetImages")  # Uncomment this line to train
    # prediction, confidence = test_image("PetImages/Cat/1.jpg")
```

Then run:

```bash
python main.py
```

#### What happens:
- Trains for **20 epochs**.
- Logs accuracy/loss per epoch.
- TensorBoard logs saved to `runs/`.
- Model saved as `cat_dog_model.pth`.

Launch TensorBoard (optional):

```bash
tensorboard --logdir=runs
```

---

### 2. Test the Model

Comment out training and use the test function instead:

```python
if __name__ == "__main__":
    # train_model("PetImages")
    prediction, confidence = test_image("PetImages/Cat/1.jpg")
    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
```

Then run:

```bash
python main.py
```

#### Output:

```bash
Prediction: Cat (Confidence: 0.94)
```

---

## 🧠 Model Architecture: `CatDogNet`

- **Input**: 64x64 RGB images
- **Layers**:
  - 3× Conv2D (with ReLU and stride-based downsampling)
  - Flatten + Fully Connected Layers
  - Final Softmax output layer with 2 neurons (Cat vs Dog)

---

## 🧪 Data Augmentation

Implemented with `albumentations`:

```python
A.Compose([
    A.Resize(64, 64),
    A.RandomCrop(64, 64),
    A.HorizontalFlip(),
    A.Rotate(limit=15),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2(),
])
```

---

## 🔍 Inference Details: `test_image(image_path)`

1. Loads `cat_dog_model.pth`.
2. Applies minimal transforms (resize + normalize).
3. Performs prediction using softmax probabilities.
4. Outputs:
   - Class: `"Cat"` or `"Dog"`
   - Confidence: Float in range `[0.0, 1.0]`

---

## 📁 File Structure

```
your_project/
├── main.py
├── cat_dog_model.pth      # Created after training
├── PetImages/
│   ├── Cat/
│   └── Dog/
└── runs/                  # TensorBoard logs
```

---

## 🛠 Customization

- 🔢 **Image Size**: Change `Resize(64, 64)` in transforms.
- 🔁 **Epoch Count**: Modify `range(20)` in the training loop.
- 🧠 **Model Depth**: Extend or simplify `CatDogNet`.
- 🎯 **Confidence Threshold**: Add post-filtering in `test_image()`.

---

## 📷 Example Prediction

```
Input Image: PetImages/Dog/12.jpg
Prediction: Dog (Confidence: 0.97)
```
