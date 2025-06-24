# Soccer Match Outcome Prediction Web App

A minimalistic Django web application that uses a machine learning model to predict soccer match outcomes based on historical data, team, league, and match stage.

---

## 🚀 Features

* **Global ML Model**: A single Random Forest classifier trained on the European Soccer Database covering all teams and leagues.
* **Interactive Form**: Users select **League**, **Home Team**, **Away Team**, and enter **Stage** via dropdowns and input fields.
* **Instant Predictions**: The backend loads the pre-trained model at startup and returns predictions in real time.
* **Clean Frontend**: Minimalistic, centered layout using Flexbox (HTML/CSS).

---

## 📦 Tech Stack

* **Backend**: Django 4.x
* **Machine Learning**: scikit-learn, pandas, numpy, joblib
* **Data Source**: European Soccer Database (SQLite)
* **Version Control**: Git + Git LFS (for large model file)
* **Frontend**: HTML5, CSS3 (Flexbox)

---

## 📁 Project Structure

```
soccer-match-prediction/
├── manage.py
├── requirements.txt
├── soccer_prediction/        # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── predictor/                # Django app
    ├── ml/
    │   └── full_model.pkl    # Pre-trained pipeline + label encoder
    ├── static/
    │   └── predictor/css/style.css  # Flexbox centering and styles
    ├── templates/
    │   └── predictor/
    │       ├── index.html    # Input form template
    │       └── result.html   # Prediction result template
    ├── views.py              # Form handling & prediction logic
    ├── urls.py               # URL routes for the app
    └── apps.py               # App configuration
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/soccer-match-prediction.git
   cd soccer-match-prediction
   ```

2. **Install Git LFS** (for large model file)

   ```bash
   # macOS (Homebrew)
   brew install git-lfs
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   # Windows: download installer from https://git-lfs.github.com/

   git lfs install
   ```

3. **Initialize the Model File**

   ```bash
   git lfs pull
   ```

4. **Create & Activate a Virtual Environment**

   ```bash
   python -m venv env
   source env/bin/activate      # On Windows: env\Scripts\activate
   ```

5. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

6. **Django Migrations**

   ```bash
   python manage.py migrate
   ```

---

## ▶️ Running the Web App

1. **Start the Development Server**

   ```bash
   python manage.py runserver
   ```

2. **Open in Browser**
   Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the predictor form.

3. **Make a Prediction**

   * Select a **League**, **Home Team**, **Away Team**, and enter the **Stage**.
   * Click **Predict Outcome** and view the result.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

