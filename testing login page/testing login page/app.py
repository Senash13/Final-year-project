from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import subprocess
import threading
import time
from bson import ObjectId

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-very-secret-key'  # Use a fixed string!

# Optional: Set the URI directly if not using .env
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://senash:1234@cluster0.moivwlp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

# MongoDB connection
client = MongoClient(MONGODB_URI)
db = client['adidas_db']        # Your new database name
users = db['admin']      

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except Exception as e:
        print("User loader error:", e)
    return None

def run_streamlit():
    # Get the path to your Streamlit app
    streamlit_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test6.py')
    subprocess.run(['streamlit', 'run', streamlit_path])

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_data = users.find_one({'username': username})
        print("User data from DB:", user_data)
        if user_data:
            print("Password in DB:", user_data['password'])
            print("Password entered:", password)
            print("Password check:", check_password_hash(user_data['password'], password))
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users.find_one({'$or': [{'username': username}, {'email': email}]}):
            flash('Username or email already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password
        }
        users.insert_one(user_data)
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Start Streamlit in a separate thread if it's not already running
    if not hasattr(app, 'streamlit_thread') or not app.streamlit_thread.is_alive():
        app.streamlit_thread = threading.Thread(target=run_streamlit)
        app.streamlit_thread.daemon = True
        app.streamlit_thread.start()
        # Give Streamlit time to start
        time.sleep(2)
    return redirect('http://localhost:8501')
    
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)