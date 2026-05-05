# 🩺 MediScan AI — Disease Prediction Platform

![MediScan AI](https://img.shields.io/badge/MediScan-AI-blue?style=for-the-badge&logo=health&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?style=for-the-badge&logo=railway&logoColor=white)

An AI-powered medical image analysis platform that detects diseases from uploaded medical images using deep learning models trained on verified open datasets.

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is not a certified medical device and should not be used as the basis for clinical diagnosis or treatment decisions. Always consult a qualified medical professional.

---

## 🌐 Live Demo

**[mediscan-ai-ds1x.onrender.com](https://mediscan-ai-ds1x.onrender.com/)**

---

## ✨ Features

- 🤖 **4 AI Disease Models** — CNN models trained on thousands of medical images
- 📤 **Image Upload & Analysis** — Drag & drop or browse to upload medical images
- 👤 **User Authentication** — Signup, login, and secure session management
- 📋 **Scan History Dashboard** — Every logged-in user's scans are saved and viewable
- ⚙️ **Admin Analytics Panel** — Charts, disease breakdowns, and platform-wide stats
- 🔓 **Guest Scanning** — Scan without login; results shown but not saved
- 💬 **Contact & Feedback Form** — Users can submit feedback with star ratings
- 📱 **Fully Responsive** — Works on desktop, tablet, and mobile

---

## 🧠 Supported Disease Models

| Disease | Input Type | Classes | Model Type |
|---|---|---|---|
| **Chest X-Ray (CXR)** | Chest X-Ray Image | COVID-19, Pneumonia, Tuberculosis, Normal | Multi-class CNN |
| **Malaria** | Blood Smear Microscopy | Parasitized, Uninfected | Binary CNN |
| **Ocular Disease** | Retinal Fundus Image | AMD, Cataract, Diabetic Retinopathy, Glaucoma, Hypertension, Myopia, Normal, Other | Multi-class CNN |
| **Brain Tumor** | MRI Scan | Tumor Detected, No Tumor | Binary CNN |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Backend** | Python, Flask |
| **ML Framework** | TensorFlow / Keras |
| **Database** | PostgreSQL (Railway) |
| **Auth** | Flask-Login, Flask-Bcrypt |
| **ORM** | Flask-SQLAlchemy |
| **Model Hosting** | Hugging Face Hub |
| **Deployment** | Railway.app |
| **Server** | Gunicorn |
| **Image Processing** | Pillow, NumPy, OpenCV |

---

## 📁 Project Structure

```
disease-predictor/
├── app.py                      # Main Flask application
├── gunicorn_config.py          # Gunicorn startup config
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for deployment
├── .python-version             # Python version pin
├── .env                        # Environment variables (not in repo)
├── .gitignore
├── templates/
│   ├── index.html              # Homepage
│   ├── login.html              # Login page
│   ├── signup.html             # Signup page
│   ├── dashboard.html          # User scan history
│   ├── admin.html              # Admin analytics
│   └── contact.html            # Contact & feedback form
├── static/
│   ├── css/
│   │   ├── style.css           # Main stylesheet
│   │   ├── auth.css            # Auth pages stylesheet
│   │   └── dashboard.css       # Dashboard stylesheet
│   └── js/
│       └── main.js             # Frontend JavaScript
└── training/
    ├── train_cxr.py            # CXR model training script
    ├── train_malaria.py        # Malaria model training script
    ├── train_ocular.py         # Ocular model training script
    └── train_brain.py          # Brain tumor model training script
```

---

## 🚀 Local Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL installed and running
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/itsRajbro/Disease-Predictor.git
cd Disease-Predictor
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create `.env` file
```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/mediscan
SECRET_KEY=your-secret-key-here
ADMIN_EMAIL=admin@mediscan.com
ADMIN_PASSWORD=Admin@1234
HF_TOKEN=your_huggingface_token
```

### Step 5 — Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🏋️ Training the Models

Datasets used (available on Kaggle):

| Model | Dataset |
|---|---|
| CXR | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| Malaria | [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |
| Ocular | [Ocular Disease Recognition ODIR5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) |
| Brain Tumor | [Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |

Place datasets in the `datasets/` folder and run:

```bash
cd training
python train_cxr.py
python train_malaria.py
python train_ocular.py
python train_brain.py
```

Trained models will be saved to the `models/` folder.

---

## ☁️ Deployment

This project is deployed on **Railway.app** with:
- **PostgreSQL** database service
- **Hugging Face Hub** for model storage (downloaded at startup)
- **Gunicorn** as the production WSGI server

### Environment Variables Required on Railway

| Key | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection URL |
| `SECRET_KEY` | Flask session secret key |
| `ADMIN_EMAIL` | Admin account email |
| `ADMIN_PASSWORD` | Admin account password |
| `HF_TOKEN` | Hugging Face API token |

---

## 👤 Default Accounts

| Role | Email | Password |
|---|---|---|
| Admin | `admin@mediscan.com` | `Admin@1234` |
| User | Register at `/signup` | Your choice |

> ⚠️ Change admin credentials before deploying to production.

---

## 📸 Pages & Routes

| Route | Description | Auth Required |
|---|---|---|
| `/` | Homepage with scan tool | No |
| `/signup` | User registration | No |
| `/login` | User login | No |
| `/dashboard` | Scan history | Yes |
| `/admin` | Admin analytics | Admin only |
| `/contact` | Contact & feedback | No |
| `/predict` | AI prediction API | No (saves if logged in) |
| `/logout` | Logout | Yes |

---

## 🔮 Roadmap

- [ ] Add more disease models (kidney, liver, dental)
- [ ] Email verification on signup
- [ ] Export scan history as PDF
- [ ] REST API with authentication tokens
- [ ] Mobile app integration

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Built with by **Ayush Raj**

[![GitHub](https://img.shields.io/badge/GitHub-itsRajbro-black?style=flat&logo=github)](https://github.com/itsRajbro)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Raj-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/www.linkedin.com/in/ayush-raj-25j07)
