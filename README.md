# 🤖 AI Image Classifier

A deep learning image classifier built with **TensorFlow** and **MobileNetV2** that can classify images into **1000 different categories** with confidence scores.

---

## 📸 What It Does

- Takes any image as input
- Uses a pre-trained **MobileNetV2** model (trained on ImageNet)
- Returns the **top 5 predictions** with confidence scores
- Displays a clean, colourful terminal interface with progress bars

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNetV2 | Pre-trained CNN model |
| NumPy | Numerical processing |
| Pillow (PIL) | Image loading & processing |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/thivekshan-engineer/image-classifier.git
cd image-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the classifier
```bash
# Option 1: Pass image path as argument
python classifier.py path/to/your/image.jpg

# Option 2: Run and enter path when prompted
python classifier.py
```

---

## 📊 Example Output

```
╔══════════════════════════════════════════════╗
║        AI IMAGE CLASSIFIER                  ║
║        Powered by TensorFlow & MobileNetV2  ║
║        by Thivekshan Rajakumar              ║
╚══════════════════════════════════════════════╝

📊 Image Info:
   • Dimensions : 640 x 480 pixels
   • Mode       : RGB
   • File size  : 85.2 KB

⏳ Loading pre-trained MobileNetV2 model...
✅ Model loaded in 3.21 seconds

🧠 Running AI classification...

──────────────────────────────────────────────────
🔍 Classification Results for: dog.jpg
──────────────────────────────────────────────────

🥇 Golden Retriever          ██████████████████░░░░░░░░░░░░ 61.24%
🥈 Labrador Retriever        ████████░░░░░░░░░░░░░░░░░░░░░░ 28.13%
🥉 Flat Coated Retriever     ███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7.45%
 4. Cocker Spaniel           █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.11%
 5. Irish Setter             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1.07%

✅ Top prediction: Golden Retriever (61.24% confidence)
```

---

## 🧠 About MobileNetV2

MobileNetV2 is a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. It was trained on the **ImageNet dataset** containing over **1.2 million images** across **1000 categories**.

---

## 📁 Project Structure

```
image-classifier/
│
├── classifier.py      # Main classifier script
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 👨‍💻 Author

**Thivekshan Rajakumar**
- 🎓 BSc Computer Systems Engineering — University of Sunderland
- 🌐 Portfolio: https://thivekshan-engineer.github.io/personal-website-/
- 💼 GitHub: https://github.com/thivekshan-engineer

---

## 📜 License

This project is open source and available under the MIT License.
