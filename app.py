from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import os
import datetime
import json
import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from functools import wraps
from sqlalchemy import func
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///confusion_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize Gemini AI
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Could not initialize Gemini AI: {e}")

# Store confusion logs
confusion_logs = []

# Global variables for model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['confused', 'neutral']

# Define image transformation
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    confusion_logs = db.relationship('ConfusionLog', backref='user', lazy=True)
    watch_history = db.relationship('VideoWatchHistory', backref='user', lazy=True)
    video_stats = db.relationship('UserVideoStats', backref='user', lazy=True)
    quizzes = db.relationship('Quiz', backref='user', lazy=True)
    roadmaps = db.relationship('Roadmap', backref='user', lazy=True)
    recommendations = db.relationship('CourseRecommendation', backref='user', lazy=True)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    youtube_id = db.Column(db.String(20), nullable=False)
    thumbnail_url = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.String(20), nullable=False)
    level = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    category = db.Column(db.String(50), default='General')
    tags = db.Column(db.String(255), default='')

class ConfusionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    timestamp = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    course = db.relationship('Course', backref='confusion_logs')

class VideoWatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    watched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    duration_watched = db.Column(db.Integer, default=0)  # in seconds
    course = db.relationship('Course', backref='watch_history')

class UserVideoStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    watch_count = db.Column(db.Integer, default=0)
    total_confusion_count = db.Column(db.Integer, default=0)
    confusion_percentage = db.Column(db.Float, default=0.0)
    last_watched = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    course = db.relationship('Course', backref='user_stats')

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    questions = db.Column(db.Text, nullable=False)  # JSON string of questions and answers
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Roadmap(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class CourseRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(200), nullable=False)
    recommendations = db.Column(db.Text, nullable=False)  # JSON string of recommendations
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Helper function to extract YouTube ID from URL
def extract_youtube_id(url):
    # Check if it's already just an ID (11 characters)
    if len(url) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    
    # Try to extract from various YouTube URL formats
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    
    return None

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('You do not have permission to access this page', 'danger')
            return redirect(url_for('index'))
        
        return f(*args, **kwargs)
    return decorated_function

def load_model():
    global model
    try:
        # Create model architecture
        model_type = 'beit_base_patch16_224'
        model = timm.create_model(model_type, pretrained=True)
        
        # Freeze first 180 layers
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 180:
                param.requires_grad = False
            else:
                break
        
        # Load your trained weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'best_model_epoch_8.pt')
        
        if not os.path.exists(weights_path):
            print(f"Warning: Model weights file not found at {weights_path}")
            print("Please place your best_model_epoch_8.pt file in the 'weights' folder")
            # Create weights directory if it doesn't exist
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Successfully loaded model weights from {weights_path}")
        
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        return False

# Gemini AI helper functions
def generate_course_recommendations(query):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        As an AI course recommendation system, suggest 5 courses based on the following query: "{query}".
        For each course, provide:
        1. Title
        2. Brief description (2-3 sentences)
        3. Difficulty level (Beginner, Intermediate, Advanced)
        4. Estimated duration
        5. Key topics covered
        
        Format your response as a JSON array with the following structure:
        [
            {{
                "title": "Course Title",
                "description": "Course description",
                "level": "Difficulty level",
                "duration": "Estimated duration",
                "topics": ["Topic 1", "Topic 2", "Topic 3"]
            }},
            ...
        ]
        
        Ensure the response is valid JSON.
        """
        
        response = model.generate_content(prompt)
        
        # Validate JSON response
        try:
            json_response = response.text
            # Test if it's valid JSON by parsing it
            json.loads(json_response)
            return json_response
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'(\[\s*{.*}\s*\])', response.text, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group(1)
                    # Validate the extracted JSON
                    json.loads(extracted_json)
                    return extracted_json
                except:
                    pass
            
            # If extraction fails, return a fallback response
            return json.dumps([
                {
                    "title": "Introduction to " + query,
                    "description": "A comprehensive introduction to " + query + " covering all the essential concepts.",
                    "level": "Beginner",
                    "duration": "8 hours",
                    "topics": ["Fundamentals", "Core concepts", "Practical applications"]
                }
            ])
    except Exception as e:
        print(f"Error generating course recommendations: {e}")
        return json.dumps([{"title": "Error", "description": f"Failed to generate recommendations: {str(e)}"}])

def generate_quiz(topic, num_questions=5):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Create a quiz on the topic of "{topic}" with {num_questions} questions.
        Each question should have 4 multiple-choice options with one correct answer.
        
        Format your response as a JSON array with the following structure:
        [
            {{
                "question": "Question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": "Brief explanation of the correct answer"
            }},
            ...
        ]
        
        Ensure the response is valid JSON.
        """
        
        response = model.generate_content(prompt)
        
        # Validate JSON response
        try:
            json_response = response.text
            # Test if it's valid JSON by parsing it
            json.loads(json_response)
            return json_response
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'(\[\s*{.*}\s*\])', response.text, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group(1)
                    # Validate the extracted JSON
                    json.loads(extracted_json)
                    return extracted_json
                except:
                    pass
            
            # If extraction fails, return a fallback response
            return json.dumps([
                {
                    "question": "What is " + topic + "?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                    "explanation": "This is a placeholder question since we couldn't generate a proper quiz."
                }
            ])
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return json.dumps([{"question": "Error", "options": ["Failed to generate quiz"], "correct_answer": "", "explanation": str(e)}])

def generate_roadmap(topic):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Create a comprehensive learning roadmap for someone who wants to learn "{topic}" from beginner to advanced level.
        
        Include the following sections:
        1. Prerequisites - What should they know before starting
        2. Beginner Level - Fundamental concepts and skills
        3. Intermediate Level - More advanced concepts and practical applications
        4. Advanced Level - Expert-level skills and specialized knowledge
        5. Recommended Resources - Books, courses, websites, and tools
        
        Format your response in Markdown with clear headings, bullet points, and brief descriptions.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating roadmap: {e}")
        return f"# Error\n\nFailed to generate roadmap: {str(e)}"

# Default route - redirect to login
@app.route('/')
def default_route():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))
    return redirect(url_for('login'))

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            password=hashed_password,
            name=name,
            age=age,
            gender=gender
        )
        
        # Add admin user if it's the first user
        if User.query.count() == 0:
            new_user.is_admin = True
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check for admin login
        if username == 'admin' and password == 'admin':
            admin_user = User.query.filter_by(is_admin=True).first()
            if admin_user:
                session['user_id'] = admin_user.id
                flash('Admin login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                # Create admin user if it doesn't exist
                hashed_password = generate_password_hash('admin')
                admin_user = User(
                    username='admin',
                    password=hashed_password,
                    name='Administrator',
                    age=0,
                    gender='Other',
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
                
                session['user_id'] = admin_user.id
                flash('Admin account created and logged in!', 'success')
                return redirect(url_for('admin_dashboard'))
        
        # Regular user login
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        session['user_id'] = user.id
        flash('Login successful!', 'success')
        
        if user.is_admin:
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Main routes
@app.route('/home')
@login_required
def index():
    courses = Course.query.order_by(Course.created_at.desc()).limit(4).all()
    current_year = 2025  # Updated year
    return render_template('index.html', courses=courses, current_year=current_year)

@app.route('/video/<int:video_id>')
@login_required
def video(video_id):
    course = Course.query.get_or_404(video_id)

    # Check if user has stats for this video
    user_stats = UserVideoStats.query.filter_by(
        user_id=session['user_id'],
        video_id=video_id
    ).first()

    # If not, create new stats
    if not user_stats:
        user_stats = UserVideoStats(
            user_id=session['user_id'],
            video_id=video_id,
            watch_count=0,
            total_confusion_count=0,
            confusion_percentage=0.0
        )
        db.session.add(user_stats)

    # Increment watch count
    user_stats.watch_count += 1
    user_stats.last_watched = datetime.datetime.utcnow()
    db.session.commit()

    # Log video watch history
    watch_history = VideoWatchHistory(
        user_id=session['user_id'],
        video_id=video_id
    )
    db.session.add(watch_history)
    db.session.commit()

    return render_template('video.html', video_id=course.youtube_id, video_num=course.id, course=course)

@app.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    global model

    # Load model if not already loaded
    if model is None:
        success = load_model()
        if not success:
            return jsonify({'error': 'Model could not be loaded'}), 500

    data = request.json
    image_data = data.get('image')
    video_time = data.get('videoTime')
    video_id = data.get('videoId')

    try:
        # Check if image data is empty or None
        if not image_data or len(image_data.split(',')) < 2:
            return jsonify({
                'status': 'no_face',
                'message': 'No face detected in the camera'
            })
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get predictions
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities, dim=0).item()
        confidence = probabilities[predicted_class].item() * 100
        predicted_class_name = class_names[predicted_class]
        
        # Format timestamp as minutes:seconds
        minutes = int(video_time // 60)
        seconds = int(video_time % 60)
        formatted_time = f"{minutes}:{seconds:02d}"
        
        # Log confusion if detected
        if predicted_class_name == 'confused':
            log_entry = ConfusionLog(
                user_id=session['user_id'],
                video_id=video_id,
                timestamp=formatted_time,
                confidence=confidence
            )
            
            db.session.add(log_entry)
            
            # Update user video stats
            user_stats = UserVideoStats.query.filter_by(
                user_id=session['user_id'],
                video_id=video_id
            ).first()
            
            if user_stats:
                user_stats.total_confusion_count += 1
                
                # Calculate new confusion percentage
                # Get total number of detections for this session
                watch_history = VideoWatchHistory.query.filter_by(
                    user_id=session['user_id'],
                    video_id=video_id
                ).order_by(VideoWatchHistory.watched_at.desc()).first()
                
                if watch_history:
                    # Get confusion logs for this session
                    session_logs = ConfusionLog.query.filter_by(
                        user_id=session['user_id'],
                        video_id=video_id
                    ).filter(ConfusionLog.detected_at >= watch_history.watched_at).count()
                    
                    # Assuming we check every 2 seconds, calculate total checks
                    total_checks = max(1, watch_history.duration_watched // 2)
                    
                    # Calculate percentage
                    confusion_percentage = (session_logs / total_checks) * 100
                    user_stats.confusion_percentage = confusion_percentage
            
            db.session.commit()
            
            # Print to terminal for demonstration
            print(f"Confusion detected at video time {formatted_time} for video {video_id}")
            print(f"Confidence: {confidence:.1f}%")
        
        return jsonify({
            'status': 'success',
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.1f}"
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_watch_duration', methods=['POST'])
@login_required
def update_watch_duration():
    data = request.json
    video_id = data.get('videoId')
    duration = data.get('duration')

    # Update the most recent watch history for this user and video
    watch_history = VideoWatchHistory.query.filter_by(
        user_id=session['user_id'],
        video_id=video_id
    ).order_by(VideoWatchHistory.watched_at.desc()).first()

    if watch_history:
        watch_history.duration_watched = duration
        db.session.commit()

    return jsonify({'status': 'success'})

@app.route('/courses')
@login_required
def courses():
    all_courses = Course.query.all()
    current_year = 2025  # Updated year
    return render_template('courses.html', courses=all_courses, current_year=current_year)

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('logout'))
        
    confusion_logs = ConfusionLog.query.filter_by(user_id=session['user_id']).order_by(ConfusionLog.detected_at.desc()).all()
    watch_history = VideoWatchHistory.query.filter_by(user_id=session['user_id']).order_by(VideoWatchHistory.watched_at.desc()).all()
    video_stats = UserVideoStats.query.filter_by(user_id=session['user_id']).all()
    
    # Fix for CourseRecommendation query
    recommendations = []
    try:
        recommendations = CourseRecommendation.query.filter_by(user_id=session['user_id']).order_by(CourseRecommendation.created_at.desc()).all()
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
    
    # Fix for Quiz query
    quizzes = []
    try:
        quizzes = Quiz.query.filter_by(user_id=session['user_id']).order_by(Quiz.created_at.desc()).all()
    except Exception as e:
        print(f"Error fetching quizzes: {e}")
    
    # Fix for Roadmap query
    roadmaps = []
    try:
        roadmaps = Roadmap.query.filter_by(user_id=session['user_id']).order_by(Roadmap.created_at.desc()).all()
    except Exception as e:
        print(f"Error fetching roadmaps: {e}")
    
    current_year = 2025  # Updated year

    return render_template('profile.html', 
                        user=user, 
                        confusion_logs=confusion_logs, 
                        watch_history=watch_history, 
                        video_stats=video_stats,
                        quizzes=quizzes,
                        roadmaps=roadmaps,
                        recommendations=recommendations,
                        current_year=current_year)

# AI-powered features
@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend_courses():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            flash('Please enter a search query', 'danger')
            return redirect(url_for('recommend_courses'))
        
        recommendations_json = generate_course_recommendations(query)
        
        # Save recommendation to database
        recommendation = CourseRecommendation(
            user_id=session['user_id'],
            query=query,
            recommendations=recommendations_json
        )
        db.session.add(recommendation)
        db.session.commit()
        
        try:
            recommendations = json.loads(recommendations_json)
        except json.JSONDecodeError as e:
            recommendations = []
            flash(f'Error parsing recommendations: {str(e)}', 'danger')
        
        current_year = 2025  # Updated year
        return render_template('recommend.html', 
                        recommendations=recommendations, 
                        query=query,
                        recommendation_id=recommendation.id,
                        current_year=current_year)

    # Handle GET request with recommendation ID
    recommendation_id = request.args.get('id')
    if recommendation_id:
        recommendation = CourseRecommendation.query.get(recommendation_id)
        if recommendation and recommendation.user_id == session['user_id']:
            try:
                recommendations = json.loads(recommendation.recommendations)
                current_year = 2025  # Updated year
                return render_template('recommend.html',
                                recommendations=recommendations,
                                query=recommendation.query,
                                recommendation_id=recommendation.id,
                                current_year=current_year)
            except json.JSONDecodeError as e:
                flash(f'Error loading recommendation: {str(e)}', 'danger')

    current_year = 2025  # Updated year
    return render_template('recommend.html', current_year=current_year)

@app.route('/quiz', methods=['GET', 'POST'])
@login_required
def generate_quiz_page():
    if request.method == 'POST':
        topic = request.form.get('topic')
        num_questions = int(request.form.get('num_questions', 5))
        
        if not topic:
            flash('Please enter a topic', 'danger')
            return redirect(url_for('generate_quiz_page'))
        
        quiz_json = generate_quiz(topic, num_questions)
        
        # Save quiz to database
        quiz = Quiz(
            user_id=session['user_id'],
            topic=topic,
            questions=quiz_json
        )
        db.session.add(quiz)
        db.session.commit()
        
        try:
            questions = json.loads(quiz_json)
        except json.JSONDecodeError as e:
            questions = []
            flash(f'Error parsing quiz questions: {str(e)}', 'danger')
        
        current_year = 2025  # Updated year
        return render_template('quiz.html', 
                        questions=questions, 
                        topic=topic,
                        quiz_id=quiz.id,
                        current_year=current_year)

    current_year = 2025  # Updated year
    return render_template('quiz.html', current_year=current_year)

@app.route('/quiz/<int:quiz_id>')
@login_required
def view_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)

    # Check if the quiz belongs to the current user
    if quiz.user_id != session['user_id'] and not User.query.get(session['user_id']).is_admin:
        flash('You do not have permission to view this quiz', 'danger')
        return redirect(url_for('profile'))

    try:
        questions = json.loads(quiz.questions)
    except json.JSONDecodeError as e:
        questions = []
        flash(f'Error parsing quiz questions: {str(e)}', 'danger')

    current_year = 2025  # Updated year
    return render_template('quiz.html', 
                    questions=questions, 
                    topic=quiz.topic,
                    quiz_id=quiz.id,
                    current_year=current_year)

@app.route('/roadmap', methods=['GET', 'POST'])
@login_required
def generate_roadmap_page():
    if request.method == 'POST':
        topic = request.form.get('topic')
        
        if not topic:
            flash('Please enter a topic', 'danger')
            return redirect(url_for('generate_roadmap_page'))
        
        roadmap_content = generate_roadmap(topic)
        
        # Save roadmap to database
        roadmap = Roadmap(
            user_id=session['user_id'],
            topic=topic,
            content=roadmap_content
        )
        db.session.add(roadmap)
        db.session.commit()
        
        current_year = 2025  # Updated year
        return render_template('roadmap.html', 
                        content=roadmap_content, 
                        topic=topic,
                        roadmap_id=roadmap.id,
                        current_year=current_year)

    current_year = 2025  # Updated year
    return render_template('roadmap.html', current_year=current_year)

@app.route('/roadmap/<int:roadmap_id>')
@login_required
def view_roadmap(roadmap_id):
    roadmap = Roadmap.query.get_or_404(roadmap_id)

    # Check if the roadmap belongs to the current user
    if roadmap.user_id != session['user_id'] and not User.query.get(session['user_id']).is_admin:
        flash('You do not have permission to view this roadmap', 'danger')
        return redirect(url_for('profile'))

    current_year = 2025  # Updated year
    return render_template('roadmap.html', 
                    content=roadmap.content, 
                    topic=roadmap.topic,
                    roadmap_id=roadmap.id,
                    current_year=current_year)

# Admin routes
@app.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.filter_by(is_admin=False).all()
    total_users = len(users)
    total_confusion_logs = ConfusionLog.query.count()
    total_videos_watched = VideoWatchHistory.query.count()
    current_year = 2025  # Updated year

    return render_template('admin_dashboard.html', 
                    users=users, 
                    total_users=total_users,
                    total_confusion_logs=total_confusion_logs,
                    total_videos_watched=total_videos_watched,
                    Course=Course,
                    current_year=current_year)

@app.route('/admin/user/<int:user_id>')
@admin_required
def admin_user_details(user_id):
    user = User.query.get_or_404(user_id)
    confusion_logs = ConfusionLog.query.filter_by(user_id=user_id).order_by(ConfusionLog.detected_at.desc()).all()
    watch_history = VideoWatchHistory.query.filter_by(user_id=user_id).order_by(VideoWatchHistory.watched_at.desc()).all()
    video_stats = UserVideoStats.query.filter_by(user_id=user_id).all()

    # Get all courses for reference
    courses = Course.query.all()
    current_year = 2025  # Updated year

    return render_template('admin_user_details.html', 
                    user=user, 
                    confusion_logs=confusion_logs, 
                    watch_history=watch_history, 
                    video_stats=video_stats,
                    courses=courses,
                    current_year=current_year)

@app.route('/admin/confusion_logs')
@admin_required
def admin_confusion_logs():
    logs = ConfusionLog.query.join(User).order_by(ConfusionLog.detected_at.desc()).all()
    current_year = 2025  # Updated year
    return render_template('admin_confusion_logs.html', logs=logs, current_year=current_year)

@app.route('/admin/video_analytics')
@admin_required
def admin_video_analytics():
    # Get video watch counts
    video_counts = db.session.query(
        VideoWatchHistory.video_id,
        db.func.count(VideoWatchHistory.id).label('count')
    ).group_by(VideoWatchHistory.video_id).all()

    # Get confusion counts per video
    confusion_counts = db.session.query(
        ConfusionLog.video_id,
        db.func.count(ConfusionLog.id).label('count')
    ).group_by(ConfusionLog.video_id).all()

    # Get all courses
    courses = Course.query.all()

    # Get user stats per video
    user_video_stats = UserVideoStats.query.all()

    # Get all users
    users = User.query.filter_by(is_admin=False).all()
    current_year = 2025  # Updated year

    return render_template('admin_video_analytics.html', 
                    video_counts=video_counts,
                    confusion_counts=confusion_counts,
                    courses=courses,
                    user_video_stats=user_video_stats,
                    users=users,
                    current_year=current_year)

@app.route('/admin/courses')
@admin_required
def admin_courses():
    courses = Course.query.all()
    current_year = 2025  # Updated year

    return render_template('admin_courses.html', courses=courses, current_year=current_year)

@app.route('/admin/add_course', methods=['GET', 'POST'])
@admin_required
def admin_add_course():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        youtube_id_or_url = request.form.get('youtube_id')
        thumbnail_url = request.form.get('thumbnail_url')
        duration = request.form.get('duration')
        level = request.form.get('level')
        category = request.form.get('category', 'General')
        tags = request.form.get('tags', '')
        
        # Extract YouTube ID if a full URL was provided
        youtube_id = extract_youtube_id(youtube_id_or_url)
        
        if not youtube_id:
            flash('Invalid YouTube URL or ID', 'danger')
            return redirect(url_for('admin_add_course'))
        
        # If thumbnail URL is not provided, generate it from YouTube ID
        if not thumbnail_url or thumbnail_url.strip() == '':
            thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
        
        # Create new course
        new_course = Course(
            title=title,
            description=description,
            youtube_id=youtube_id,
            thumbnail_url=thumbnail_url,
            duration=duration,
            level=level,
            category=category,
            tags=tags
        )
        
        db.session.add(new_course)
        db.session.commit()
        
        flash('Course added successfully!', 'success')
        return redirect(url_for('admin_courses'))

    current_year = 2025  # Updated year
    return render_template('admin_add_course.html', current_year=current_year)

@app.route('/admin/edit_course/<int:course_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_course(course_id):
    course = Course.query.get_or_404(course_id)

    if request.method == 'POST':
        course.title = request.form.get('title')
        course.description = request.form.get('description')
        youtube_id_or_url = request.form.get('youtube_id')
        course.thumbnail_url = request.form.get('thumbnail_url')
        course.duration = request.form.get('duration')
        course.level = request.form.get('level')
        course.category = request.form.get('category', 'General')
        course.tags = request.form.get('tags', '')
        
        # Extract YouTube ID if a full URL was provided
        youtube_id = extract_youtube_id(youtube_id_or_url)
        
        if not youtube_id:
            flash('Invalid YouTube URL or ID', 'danger')
            return redirect(url_for('admin_edit_course', course_id=course_id))
        
        course.youtube_id = youtube_id
        
        # If thumbnail URL is not provided, generate it from YouTube ID
        if not course.thumbnail_url or course.thumbnail_url.strip() == '':
            course.thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
        
        db.session.commit()
        
        flash('Course updated successfully!', 'success')
        return redirect(url_for('admin_courses'))

    current_year = 2025  # Updated year
    return render_template('admin_edit_course.html', course=course, current_year=current_year)

@app.route('/admin/delete_course/<int:course_id>', methods=['POST'])
@admin_required
def admin_delete_course(course_id):
    course = Course.query.get_or_404(course_id)

    # Delete related records
    ConfusionLog.query.filter_by(video_id=course_id).delete()
    VideoWatchHistory.query.filter_by(video_id=course_id).delete()
    UserVideoStats.query.filter_by(video_id=course_id).delete()

    db.session.delete(course)
    db.session.commit()

    flash('Course deleted successfully!', 'success')
    return redirect(url_for('admin_courses'))

@app.route('/admin/user_tracking')
@admin_required
def admin_user_tracking():
    users = User.query.filter_by(is_admin=False).all()
    courses = Course.query.all()
    user_video_stats = UserVideoStats.query.all()
    current_year = 2025  # Updated year

    return render_template('admin_user_tracking.html', 
                    users=users,
                    courses=courses,
                    user_video_stats=user_video_stats,
                    current_year=current_year)

if __name__ == '__main__':
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'weights'), exist_ok=True)

    # Create database if it doesn't exist
    with app.app_context():
        db.create_all()
        
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            hashed_password = generate_password_hash('admin')
            admin = User(
                username='admin',
                password=hashed_password,
                name='Administrator',
                age=0,
                gender='Other',
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
        
        # Add default courses if none exist
        if Course.query.count() == 0:
            default_courses = [
                {
                    'title': 'Introduction to Machine Learning',
                    'description': 'Learn the basics of machine learning and its applications in the real world.',
                    'youtube_id': 'lhNOi5q8BF4',
                    'thumbnail_url': 'https://img.youtube.com/vi/lhNOi5q8BF4/maxresdefault.jpg',
                    'duration': '45 minutes',
                    'level': 'Beginner',
                    'category': 'Data Science',
                    'tags': 'machine learning, AI, data science'
                },
                {
                    'title': 'Advanced Python Programming',
                    'description': 'Take your Python skills to the next level with advanced techniques and patterns.',
                    'youtube_id': 'd2kxUVwWWwU',
                    'thumbnail_url': 'https://img.youtube.com/vi/d2kxUVwWWwU/maxresdefault.jpg',
                    'duration': '60 minutes',
                    'level': 'Intermediate',
                    'category': 'Programming',
                    'tags': 'python, programming, advanced'
                },
                {
                    'title': 'Web Development Fundamentals',
                    'description': 'Learn the core concepts of modern web development with HTML, CSS, and JavaScript.',
                    'youtube_id': 'zfiSAzpy9NM',
                    'thumbnail_url': 'https://img.youtube.com/vi/zfiSAzpy9NM/maxresdefault.jpg',
                    'duration': '55 minutes',
                    'level': 'Beginner',
                    'category': 'Web Development',
                    'tags': 'web, html, css, javascript'
                },
                {
                    'title': 'Data Science Essentials',
                    'description': 'Discover the essential skills and tools needed for data science and analytics.',
                    'youtube_id': 'q6kJ71tEYqM',
                    'thumbnail_url': 'https://img.youtube.com/vi/q6kJ71tEYqM/maxresdefault.jpg',
                    'duration': '50 minutes',
                    'level': 'Intermediate',
                    'category': 'Data Science',
                    'tags': 'data science, analytics, statistics'
                }
            ]
            
            for course_data in default_courses:
                course = Course(**course_data)
                db.session.add(course)
            
            db.session.commit()

    # Load model at startup
    load_model()
    app.run(debug=True)

