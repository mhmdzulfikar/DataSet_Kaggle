

### ✅ **Versi yang Sudah Diperbaiki & Dioptimalkan**

```markdown
## 📊 Customer Churn Prediction with Machine Learning

This project applies classification algorithms to predict customer churn using behavioral and demographic features.

---

### 📁 Folder Structure

```

.KAGGLE/
├── data/
│   ├── customer\_churn\_dataset-training-master.csv
│   ├── customer\_churn\_dataset-testing-master.csv
├── main.py
├── DiagnosticAnalytics.py
├── PredictiveAnalytics.py
├── ExcelUsabilityChart.py
└── kaggle.json

````

---

### ⚙️ How to Run Locally

#### ✅ 1. Clone the Repository

```bash
git clone https://github.com/mhmdzulfikar/DataSet_Kaggle.git
cd DataSet_Kaggle/.KAGGLE
````

#### ✅ 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv env
# For Linux/macOS
source env/bin/activate
# For Windows
env\Scripts\activate
```

#### ✅ 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` not available, install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

#### ✅ 4. Prepare Dataset

Ensure the following files exist in the `data/` folder:

* `customer_churn_dataset-training-master.csv`
* `customer_churn_dataset-testing-master.csv`

If you don't have them, download from [Kaggle Dataset Input](https://www.kaggle.com/code/sohailaelsayed/customer-churn-eda-ml/input)

Create the folder manually if missing:

```bash
mkdir data
```

#### ✅ 5. Run the Script

```bash
python main.py
```

---

### 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

To export the exact versions used in your environment:

```bash
pip freeze > requirements.txt
```

---

### 🤖 Models Used

* ✅ Logistic Regression
* 🌳 Decision Tree Classifier
* 🌲 Random Forest Classifier

All models include:

* Accuracy score
* Classification report
* Confusion matrix visualized with `seaborn`

---

### 📈 Extra Visualizations

| File                     | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `DiagnosticAnalytics.py` | Visualize metrics with boxplot                   |
| `PredictiveAnalytics.py` | Regression modeling with scatterplot & trendline |
| `ExcelUsabilityChart.py` | Bar chart of Excel use across analytics stages   |

---

### 📬 License

This project is open-source and free to use for educational purposes. Attribution is appreciated.

```

---

### 🔍 Catatan Tambahan:
- Kamu **tidak perlu** menggabungkan `pip install -r requirements.txt` dan `pip install ...` jika file `requirements.txt` sudah ada.
- Kalau kamu mau lebih advanced, kamu bisa tambahkan badge (seperti versi Python, license, stars repo, dll) di bagian paling atas README.

Kalau kamu ingin saya bantu buatin file `requirements.txt` otomatis dari script-mu juga bisa!
```
