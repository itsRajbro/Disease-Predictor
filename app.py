import os, io, json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import tensorflow as tf
import numpy as np
from PIL import Image
from functools import wraps
import sys

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY']           = os.getenv('SECRET_KEY', 'fallback-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///mediscan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db           = SQLAlchemy(app)
bcrypt       = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

ADMIN_EMAIL    = os.getenv('ADMIN_EMAIL', 'admin@mediscan.com')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'Admin@1234')

# ── Models ────────────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(120), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    is_admin   = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans      = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    __tablename__ = 'scans'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    disease_type = db.Column(db.String(50), nullable=False)
    prediction   = db.Column(db.String(200), nullable=False)
    confidence   = db.Column(db.Float, nullable=False)
    scanned_at   = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Admin decorator ───────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ── Load ML models ────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download, login

HF_REPO   = "YOUR_HF_USERNAME/mediscan-models"
MODEL_DIR = "/tmp/models"

def download_models():
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = [
        "cxr_model.h5", "malaria_model.h5",
        "ocular_model.h5", "brain_model.h5",
        "cxr_classes.json", "malaria_classes.json",
        "ocular_classes.json", "brain_classes.json",
    ]
    for filename in files:
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f"⬇️ Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=MODEL_DIR,
                token=hf_token
            )
            print(f"✅ {filename} ready")

download_models()

MODELS = {}
MODEL_FILES = {
    "cxr":     f"{MODEL_DIR}/cxr_model.h5",
    "malaria":  f"{MODEL_DIR}/malaria_model.h5",
    "ocular":   f"{MODEL_DIR}/ocular_model.h5",
    "brain":    f"{MODEL_DIR}/brain_model.h5",
}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        MODELS[name] = tf.keras.models.load_model(path)
        print(f"✅ Loaded {name} model")
LABELS = {
    "cxr":    {0: "COVID-19", 1: "Normal", 2: "Pneumonia", 3: "Tuberculosis"},
    "malaria": {0: "Malaria Detected (Parasitized)", 1: "Normal (Uninfected)"},
    "ocular":  {
        0: "Age-related Macular Degeneration",
        1: "Cataract",
        2: "Diabetic Retinopathy",
        3: "Glaucoma",
        4: "Hypertension",
        5: "Myopia",
        6: "Normal",
        7: "Other"
    },
    "brain":  {0: "No Tumor Detected", 1: "Tumor Detected"},
}

def preprocess_image(image_bytes, target_size=(150, 150)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes: Pages ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([name, email, password, confirm]):
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('signup.html')

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        is_admin = (email == ADMIN_EMAIL)
        user = User(name=name, email=email, password=hashed, is_admin=is_admin)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f'Welcome, {name}!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    scans = Scan.query.filter_by(user_id=current_user.id)\
                      .order_by(Scan.scanned_at.desc()).all()
    
    normal_keywords = ['normal', 'uninfected', 'no tumor']
    normal_count = sum(
        1 for s in scans 
        if any(k in s.prediction.lower() for k in normal_keywords)
    )
    detected_count = len(scans) - normal_count

    return render_template('dashboard.html', 
        scans=scans,
        normal_count=normal_count,
        detected_count=detected_count
    )
    
@app.route('/admin')
@login_required
@admin_required
def admin():
    total_users  = User.query.count()
    total_scans  = Scan.query.count()
    recent_scans = Scan.query.order_by(Scan.scanned_at.desc()).limit(20).all()

    # Disease breakdown
    disease_counts = {}
    for name in MODELS.keys():
        disease_counts[name] = Scan.query.filter_by(disease_type=name).count()

    # Daily scans (last 7 days)
    from sqlalchemy import func
    from datetime import timedelta
    today = datetime.utcnow().date()
    daily = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Scan.query.filter(
            func.date(Scan.scanned_at) == day
        ).count()
        daily.append({'date': day.strftime('%b %d'), 'count': count})

    return render_template('admin.html',
        total_users=total_users,
        total_scans=total_scans,
        recent_scans=recent_scans,
        disease_counts=disease_counts,
        daily=daily
    )

# ── Route: Predict ────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'disease' not in request.form:
        return jsonify({'error': 'Image and disease type required'}), 400

    disease = request.form['disease'].lower()
    if disease not in MODELS:
        return jsonify({'error': f"Model for '{disease}' not loaded"}), 400

    image_bytes = request.files['image'].read()
    processed   = preprocess_image(image_bytes)
    model       = MODELS[disease]
    prediction  = model.predict(processed)
    labels      = LABELS[disease]

    if disease in ['malaria', 'brain']:
        confidence = float(prediction[0][0])
        if confidence > 0.5:
            idx = 1; confidence_pct = confidence
        else:
            idx = 0; confidence_pct = 1 - confidence
    else:
        idx = int(np.argmax(prediction[0]))
        confidence_pct = float(prediction[0][idx])

    label = labels[idx]
    confidence_rounded = round(confidence_pct * 100, 2)

    # ── Only save to DB if user is logged in ──────────────────────────────────
    if current_user.is_authenticated:
        scan = Scan(
            user_id=current_user.id,
            disease_type=disease,
            prediction=label,
            confidence=confidence_rounded
        )
        db.session.add(scan)
        db.session.commit()

    return jsonify({
        'disease_type': disease,
        'prediction':   label,
        'confidence':   confidence_rounded,
        'saved':        current_user.is_authenticated
    })
# ── Temporary model upload route (remove after deployment) ───────────────────
@app.route('/upload-model', methods=['POST'])
def upload_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    file = request.files['model']
    file.save(os.path.join(MODEL_DIR, file.filename))
    return jsonify({'saved': file.filename})
# ── Init DB + Admin ───────────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        # Create hardcoded admin if not exists
        if not User.query.filter_by(email=ADMIN_EMAIL).first():
            hashed = bcrypt.generate_password_hash(ADMIN_PASSWORD).decode('utf-8')
            admin_user = User(
                name='Admin',
                email=ADMIN_EMAIL,
                password=hashed,
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print(f"✅ Admin created: {ADMIN_EMAIL}")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)