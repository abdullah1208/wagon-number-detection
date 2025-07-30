# 🚆 Automated Train Wagon Detection and Number Reader (Streamlit App)

This project is a complete **streamlit-based dashboard** that uses **YOLO object detection** and **Tesseract OCR** to detect and extract wagon numbers from uploaded train videos. It processes the video, detects individual wagons, extracts the number regions, performs OCR, and displays the results in a visual and interactive dashboard.

---

## 🎯 Features

- Detects freight **wagons** and **wagon number regions** from video
- Tracks wagons using DeepSORT for multiple object tracking
- Reads wagon numbers using Tesseract OCR
- Shows live bounding boxes and progress in Streamlit
- Displays results in a structured dashboard: wagon image, number image, OCR result, direction, and timestamp

---

## 🗂 Project Structure

```
wagon-number-reader/
├── app.py                  # Streamlit app
├── processing.py           # Detection, OCR, tracking logic
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── models/
│   ├── segment.pt          # Wagon detection model
│   └── wagon_number.pt     # Wagon number region model
├── saved_wagons/           # Output wagon images
├── saved_numbers/          # Output number images
└── sample_data/            # Optional: test video
```

---

## 🚀 How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/wagon-number-reader.git
cd wagon-number-reader
```

### 2. (Optional) Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/macOS
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🔧 Tesseract OCR Setup

This project uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for reading wagon numbers.

### 📦 Windows:
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add the path in `processing.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 🐧 Linux/macOS:
```bash
sudo apt install tesseract-ocr
# or
brew install tesseract
```

---

## 📂 Dataset Source

This project uses two custom YOLO-format datasets:

1. **Wagon Number Region Detection Dataset**  
   📎 [Train Number Detection Dataset on Kaggle](https://www.kaggle.com/datasets/abdullahgour/train-number-detection-dataset)

2. **Freight Wagon Detection Dataset**  
   📎 [Dataset for Wagon Detection of Freight Trains on Kaggle](https://www.kaggle.com/datasets/abdullahgour/dataset-for-wagon-detection-of-freight-trains)

These datasets were created, annotated, and published by the author for this project.

---

## 📦 Requirements

```
streamlit
ultralytics
opencv-python
pytesseract
numpy
Pillow
pandas
deep_sort_realtime
```

---

## 📷 Sample Output

- ✅ Detected wagon image
- ✅ Detected wagon number region
- ✅ OCR result
- ✅ Movement direction
- ✅ Timestamp

---

## 👨‍💻 Author

**Abdullah **  
AI Engineer | Computer Vision | OCR | YOLO | Streamlit Dashboard Creator

---

## 📄 License

This project is licensed under the **MIT License** – feel free to use and contribute!
