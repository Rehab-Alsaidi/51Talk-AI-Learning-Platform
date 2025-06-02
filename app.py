from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import sqlite3
import os
import secrets
import csv
import random  # Added for motivational messages
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
from functools import wraps
import tempfile
import zipfile
import io

# Load environment variables
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

# Configuration
DB_NAME = 'database.db'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'ppt', 'pptx', 'doc', 'docx'}
ACCESS_PASSWORD = "5151"  # Change this in production
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported languages
LANGUAGES = {
    'en': 'English',
    'zh': '中文',
    'ar': 'العربية'
}

# Translation dictionaries
TRANSLATIONS = {
    'en': {
        'welcome': 'Welcome to the 51Talk AI Learning Platform',
        'login': 'Login',
        'register': 'Register',
        'username': 'Username',
        'password': 'Password',
        'submit': 'Submit',
        'dashboard': 'Dashboard',
        'ai_chat': 'AI Chat',
        'learning': 'Learning',
        'progress': 'Progress',
        'settings': 'Settings',
        'logout': 'Logout',
        'welcome_msg': 'Welcome to the 51Talk AI Learning Platform!',
        'continue_learning': 'Continue Learning',
        'your_progress': 'Your Progress',
        'unit': 'Unit',
        'completed': 'Completed',
        'score': 'Score',
        'ask_ai': 'Ask the AI',
        'chat_here': 'Type your message here',
        'send': 'Send',
        'learning_materials': 'Learning Materials',
        'quizzes': 'Quizzes',
        'videos': 'Videos',
        'projects': 'Projects',
        'feedback': 'Feedback',
        'account': 'Account',
        'language': 'Language',
        'save': 'Save',
        # New keys for enhanced translation
        'learn_unit_desc': 'Learn essential concepts and practice with interactive exercises.',
        'review_unit': 'Review Unit',
        'start_unit': 'Start Unit',
        'locked': 'Locked',
        'units_completed': 'Units Completed',
        'units_remaining': 'Units Remaining',
        'daily_tip': 'Daily Tip',
        'submit_feedback': 'Submit Feedback',
        'your_feedback': 'Your Feedback',
        'rating': 'Rating',
        'excellent': 'Excellent',
        'good': 'Good',
        'average': 'Average',
        'fair': 'Fair',
        'poor': 'Poor',
        'ai_learning_assistant': 'AI Learning Assistant',
        'ask_course_question': 'Ask any question about the course material',
        'your_question': 'Your question',
        'ask_ai_placeholder': 'Ask anything about AI concepts...',
        'ask_assistant': 'Ask Assistant',
        'assistant_response': 'Assistant\'s Response',
        'ask_another_question': 'Ask Another Question',
    },
    'zh': {
        'welcome': '欢迎来到51Talk人工智能学习平台',
        'login': '登录',
        'register': '注册',
        'username': '用户名',
        'password': '密码',
        'submit': '提交',
        'dashboard': '仪表板',
        'ai_chat': 'AI聊天',
        'learning': '学习',
        'progress': '进度',
        'settings': '设置',
        'logout': '退出',
        'welcome_msg': '欢迎来到51Talk人工智能学习平台！',
        'continue_learning': '继续学习',
        'your_progress': '您的进度',
        'unit': '单元',
        'completed': '已完成',
        'score': '分数',
        'ask_ai': '问AI',
        'chat_here': '在这里输入您的消息',
        'send': '发送',
        'learning_materials': '学习材料',
        'quizzes': '小测验',
        'videos': '视频',
        'projects': '项目',
        'feedback': '反馈',
        'account': '账户',
        'language': '语言',
        'save': '保存',
        # New keys for enhanced translation
        'learn_unit_desc': '学习基本概念并通过互动练习进行练习。',
        'review_unit': '复习单元',
        'start_unit': '开始单元',
        'locked': '已锁定',
        'units_completed': '已完成单元',
        'units_remaining': '剩余单元',
        'daily_tip': '每日提示',
        'submit_feedback': '提交反馈',
        'your_feedback': '您的反馈',
        'rating': '评分',
        'excellent': '优秀',
        'good': '良好',
        'average': '一般',
        'fair': '尚可',
        'poor': '差',
        'ai_learning_assistant': 'AI学习助手',
        'ask_course_question': '提出任何关于课程材料的问题',
        'your_question': '您的问题',
        'ask_ai_placeholder': '询问任何关于AI概念的问题...',
        'ask_assistant': '询问助手',
        'assistant_response': '助手的回答',
        'ask_another_question': '提出另一个问题',
    },
    'ar': {
        'welcome': 'مرحباً بك في منصة 51Talk للتعلم بالذكاء الاصطناعي',
        'login': 'تسجيل الدخول',
        'register': 'التسجيل',
        'username': 'اسم المستخدم',
        'password': 'كلمة المرور',
        'submit': 'إرسال',
        'dashboard': 'اللوحة الرئيسية',
        'ai_chat': 'محادثة الذكاء الاصطناعي',
        'learning': 'التعلم',
        'progress': 'التقدم',
        'settings': 'الإعدادات',
        'logout': 'تسجيل الخروج',
        'welcome_msg': 'مرحباً بك في منصة 51Talk للتعلم بالذكاء الاصطناعي!',
        'continue_learning': 'متابعة التعلم',
        'your_progress': 'تقدمك',
        'unit': 'الوحدة',
        'completed': 'مكتمل',
        'score': 'الدرجة',
        'ask_ai': 'اسأل الذكاء الاصطناعي',
        'chat_here': 'اكتب رسالتك هنا',
        'send': 'إرسال',
        'learning_materials': 'المواد التعليمية',
        'quizzes': 'الاختبارات',
        'videos': 'الفيديوهات',
        'projects': 'المشاريع',
        'feedback': 'التعليقات',
        'account': 'الحساب',
        'language': 'اللغة',
        'save': 'حفظ',
        # New keys for enhanced translation
        'learn_unit_desc': 'تعلم المفاهيم الأساسية وتدرب باستخدام التمارين التفاعلية.',
        'review_unit': 'مراجعة الوحدة',
        'start_unit': 'بدء الوحدة',
        'locked': 'مقفل',
        'units_completed': 'الوحدات المكتملة',
        'units_remaining': 'الوحدات المتبقية',
        'daily_tip': 'نصيحة اليوم',
        'submit_feedback': 'إرسال تعليق',
        'your_feedback': 'تعليقك',
        'rating': 'التقييم',
        'excellent': 'ممتاز',
        'good': 'جيد',
        'average': 'متوسط',
        'fair': 'مقبول',
        'poor': 'ضعيف',
        'ai_learning_assistant': 'مساعد التعلم بالذكاء الاصطناعي',
        'ask_course_question': 'اطرح أي سؤال حول مواد الدورة',
        'your_question': 'سؤالك',
        'ask_ai_placeholder': 'اسأل أي شيء عن مفاهيم الذكاء الاصطناعي...',
        'ask_assistant': 'اسأل المساعد',
        'assistant_response': 'رد المساعد',
        'ask_another_question': 'اطرح سؤالاً آخر',
    }
}

# Initialize Gemini
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )
except Exception as e:
    print(f"Error initializing Gemini: {str(e)}")
    llm = None

# ---------- BEFORE REQUEST ----------
@app.before_request
def before_request():
    """Ensure language is set before each request"""
    # Don't override language if it's already set in the session
    if 'language' not in session:
        if 'username' in session:
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT language FROM users WHERE name=?", (session['username'],))
                    result = cursor.fetchone()
                    if result and result[0]:
                        session['language'] = result[0]
                        print(f"Setting language to {result[0]} from database")
                    else:
                        session['language'] = 'en'
                        print(f"No language found for user, defaulting to 'en'")
            except Exception as e:
                print(f"Error retrieving language from database: {str(e)}")
                session['language'] = 'en'
        else:
            session['language'] = 'en'
            print("No user logged in, defaulting to 'en'")

# ---------- AFTER REQUEST ----------
@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# ---------- CONTEXT PROCESSOR ----------
@app.context_processor
def inject_globals():
    """Make important variables available to all templates"""
    return {
        'LANGUAGES': LANGUAGES,
        'get_text': get_text,
        'current_language': session.get('language', 'en')
    }

# ---------- DATABASE UTILITIES ----------
def get_tables():
    """Get all table names from the database"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        return [table[0] for table in tables]

def get_table_columns(table_name):
    """Get column names for a specific table"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        return [col[1] for col in columns]

def check_and_create_words_table():
    """Ensure words table exists with proper columns"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Check if words table exists
        tables = get_tables()
        if 'words' not in tables:
            # Create new words table
            cursor.execute('''CREATE TABLE words (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            word TEXT,
                            definition TEXT,
                            example TEXT,
                            section INTEGER DEFAULT 1)''')
            conn.commit()
            print("Created words table")
            return True
        return True

def reset_database():
    """Delete all data from the database but keep the tables"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        try:
            # Delete all user data
            cursor.execute("DELETE FROM users")
            
            # Delete all content
            cursor.execute("DELETE FROM quizzes")
            cursor.execute("DELETE FROM projects")
            cursor.execute("DELETE FROM materials")
            cursor.execute("DELETE FROM videos")
            
            # Check if words table exists before trying to delete
            if 'words' in get_tables():
                cursor.execute("DELETE FROM words")  # Added for words table
            
            # Delete all progress
            cursor.execute("DELETE FROM progress")
            cursor.execute("DELETE FROM quiz_attempts")
            cursor.execute("DELETE FROM submissions")
            
            # Delete user activity
            cursor.execute("DELETE FROM qa_history")
            cursor.execute("DELETE FROM feedback")
            
            # Delete all files in the upload folder
            for file in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            
            # Keep admin users, just reset the admin user
            cursor.execute("DELETE FROM admin_users WHERE username != 'admin'")
            
            # Reset admin user password
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            hashed_password = generate_password_hash(admin_password)
            cursor.execute("UPDATE admin_users SET password = ? WHERE username = 'admin'", (hashed_password,))
            
            conn.commit()
            print("Database reset successfully")
            
            # Re-add admin if needed
            check_and_fix_admin_table()
            
            return True
        except sqlite3.Error as e:
            print(f"Error resetting database: {e}")
            return False

def check_and_fix_admin_table():
    """Ensure admin table has proper data"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Check if admin_users table exists
        tables = get_tables()
        if 'admin_users' not in tables:
            cursor.execute('''CREATE TABLE admin_users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            password TEXT)''')
        
        # Check if admin user exists and has proper password
        cursor.execute("SELECT id, username, password FROM admin_users WHERE username='admin'")
        admin = cursor.fetchone()
        
        # If admin doesn't exist, or password is NULL/empty, recreate it
        if not admin or not admin[2]:
            if admin:  # If exists but no proper password
                cursor.execute("DELETE FROM admin_users WHERE username='admin'")
            
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            hashed_password = generate_password_hash(admin_password)
            cursor.execute("INSERT INTO admin_users (username, password) VALUES (?, ?)", 
                         ('admin', hashed_password))
            conn.commit()
            print("Admin user recreated with default password")
        
def check_and_fix_feedback_table():
    """Ensure feedback table has proper columns"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Check if feedback table exists
        tables = get_tables()
        if 'feedback' not in tables:
            # Create new feedback table
            cursor.execute('''CREATE TABLE feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            feedback_text TEXT,
                            rating INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
            conn.commit()
            return
        
        # Check if feedback_text column exists
        columns = get_table_columns('feedback')
        
        if 'feedback_text' not in columns:
            # Rename the old table
            cursor.execute("ALTER TABLE feedback RENAME TO feedback_old")
            
            # Create new table with correct columns
            cursor.execute('''CREATE TABLE feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            feedback_text TEXT,
                            rating INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
            
            # Try to migrate data if possible
            try:
                cursor.execute('''INSERT INTO feedback (id, user_id, rating, created_at)
                                SELECT id, user_id, rating, created_at FROM feedback_old''')
            except:
                pass
                
            # Drop old table
            cursor.execute("DROP TABLE feedback_old")
            conn.commit()
            print("Fixed feedback table structure")

# ---------- HELPER FUNCTIONS ----------
def get_text(key):
    """Get translated text based on current language"""
    lang = session.get('language', 'en')
    # First try to get text from translations
    text = TRANSLATIONS.get(lang, {}).get(key)
    # If not found in current language, fall back to English
    if text is None:
        text = TRANSLATIONS.get('en', {}).get(key)
    # If still not found, return the key itself as fallback
    if text is None:
        text = key
    return text

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_id(username):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE name=?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None

def get_progress(user_id, unit_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT completed, quiz_score, project_completed FROM progress WHERE user_id=? AND unit_number=?", 
                      (user_id, unit_id))
        result = cursor.fetchone()
        return result if result else (0, 0, 0)

def has_attempted_quiz(user_id, unit_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM quiz_attempts WHERE user_id=? AND unit_id=?", (user_id, unit_id))
        return cursor.fetchone()[0] > 0

def get_admin_stats():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        stats = {}
        
        # User counts
        cursor.execute("SELECT COUNT(*) FROM users")
        stats['total_users'] = cursor.fetchone()[0]
        
        # Content counts
        cursor.execute("SELECT COUNT(*) FROM quizzes")
        stats['total_quizzes'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM materials")
        stats['total_materials'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos")
        stats['total_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM projects")
        stats['total_projects'] = cursor.fetchone()[0]
        
        # Submissions count
        cursor.execute("SELECT COUNT(*) FROM submissions")
        stats['total_submissions'] = cursor.fetchone()[0]
        
        # Recent activity
        try:
            cursor.execute("SELECT COUNT(*) FROM qa_history WHERE date(created_at) = date('now')")
            stats['today_qa'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM quiz_attempts WHERE date(attempted_at) = date('now')")
            stats['today_quiz_attempts'] = cursor.fetchone()[0]
        except sqlite3.Error:
            stats['today_qa'] = 0
            stats['today_quiz_attempts'] = 0
        
        return stats

def generate_csv_file(data, filename, headers=None):
    """Create a CSV file from data"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    
    try:
        writer = csv.writer(temp_file)
        
        # Write headers if provided
        if headers:
            writer.writerow(headers)
        
        # Write data rows
        for row in data:
            writer.writerow(row)
            
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        print(f"Error generating CSV: {str(e)}")
        return None

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin'):
            flash('Admin access required', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------- DATABASE SETUP ----------
def init_db(reset=False):
    """Initialize the database and optionally reset all data"""
    if reset:
        reset_database()
        
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Check if language column exists in users table
        try:
            cursor.execute("PRAGMA table_info(users)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Create or alter users table
            if 'users' not in get_tables():
                # User tables - create from scratch with language column
                cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    name TEXT UNIQUE,
                                    password TEXT,
                                    language TEXT DEFAULT 'en')''')
            elif 'language' not in columns:
                # Add language column if it doesn't exist
                cursor.execute("ALTER TABLE users ADD COLUMN language TEXT DEFAULT 'en'")
        except sqlite3.Error as e:
            print(f"Error setting up users table: {str(e)}")

        # Progress tracking
        cursor.execute('''CREATE TABLE IF NOT EXISTS progress (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            unit_number INTEGER,
                            completed INTEGER DEFAULT 0,
                            quiz_score INTEGER DEFAULT 0,
                            project_completed INTEGER DEFAULT 0,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Ensure unique constraint for user_id and unit_number combination
        try:
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_unit ON progress(user_id, unit_number)")
        except sqlite3.Error:
            pass
        
        # Content tables
        cursor.execute('''CREATE TABLE IF NOT EXISTS quizzes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            question TEXT,
                            options TEXT,
                            correct_answer INTEGER,
                            explanation TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS projects (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            title TEXT,
                            description TEXT,
                            resources TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS materials (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            title TEXT,
                            content TEXT,
                            file_path TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS videos (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            title TEXT,
                            youtube_url TEXT,
                            description TEXT)''')
        
        # Add words table with section column
        cursor.execute('''CREATE TABLE IF NOT EXISTS words (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            unit_id INTEGER,
                            word TEXT,
                            definition TEXT,
                            example TEXT,
                            section INTEGER DEFAULT 1)''')
        
        # Activity tracking
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            question TEXT,
                            answer TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS quiz_attempts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            unit_id INTEGER,
                            score INTEGER,
                            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS submissions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            unit_id INTEGER,
                            file_path TEXT,
                            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Feedback system - with corrected schema
        cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            feedback_text TEXT,
                            rating INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Admin tables
        cursor.execute('''CREATE TABLE IF NOT EXISTS admin_users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            password TEXT)''')
        
    # After creating tables, check and fix any issues
    check_and_fix_admin_table()
    check_and_fix_feedback_table()
    check_and_create_words_table()

# Add sample data
def add_sample_data():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Sample quiz data - removing this as we'll add through the admin interface
        pass
        
        # Sample project - removing this as we'll add through the admin interface  
        pass
        
        # Sample video - removing this as we'll add through the admin interface
        pass
        
        # Sample material - removing this as we'll add through the admin interface
        pass

        conn.commit()

# ---------- ROUTES ----------
@app.route('/', methods=['GET', 'POST'])
def password_gate():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ACCESS_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('home'))
        else:
            flash('Incorrect password. Please try again.', 'error')
    return render_template('password_gate.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if request.method == 'POST':
        name = request.form['username']
        if not name.strip():
            flash('Username cannot be empty', 'error')
            return redirect(url_for('home'))
        
        session['username'] = name
        # Clear language first to avoid conflicts
        session.pop('language', None)
        session['language'] = 'en'  # Default language
        print(f"Home: Setting default language 'en' for user {name}")
        
        with sqlite3.connect(DB_NAME) as conn:
            try:
                cursor = conn.cursor()
                # Check if user exists and get language preference
                cursor.execute("SELECT id, language FROM users WHERE name=?", (name,))
                user = cursor.fetchone()
                
                if user:
                    if user[1]:  # If language exists in database
                        session['language'] = user[1]
                        print(f"Home: Found language '{user[1]}' in database for user {name}")
                    else:
                        # Update language in database if it's null
                        cursor.execute("UPDATE users SET language = 'en' WHERE name = ?", (name,))
                        conn.commit()
                else:
                    # Create new user
                    cursor.execute("INSERT INTO users (name, language) VALUES (?, ?)", (name, 'en'))
                    conn.commit()
                    print(f"Home: Created new user {name} with language 'en'")
                
                # Force the session to be saved
                session.modified = True
                
                return redirect(url_for('dashboard'))
            except sqlite3.Error as e:
                flash(f'Error connecting to user: {str(e)}', 'error')
    
    return render_template('index.html')
@app.route('/download_material/<path:filename>')
def download_material(filename):
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    try:
        # Security check to prevent directory traversal
        safe_filename = filename.replace('../', '').replace('..\\', '')
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(request.referrer or url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    user_id = get_user_id(username)
    
    # Use current language from session, not from database
    current_language = session.get('language', 'en')
    print(f"Dashboard: Using language {current_language} for user {username}")
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT unit_number) FROM progress WHERE user_id=? AND completed=1", (user_id,))
        completed_units = cursor.fetchone()[0] or 0
    
    return render_template('dashboard.html', 
                         username=username, 
                         completed_units=completed_units,
                         current_language=current_language)  # Pass directly to template

@app.route('/logout')
def logout():
    """Handle user logout for both regular users and admin users"""
    session.pop('authenticated', None)
    session.pop('username', None)
    session.pop('language', None)
    session.pop('admin', None)
    session.pop('admin_username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('password_gate'))

@app.route('/set_language/<language>')
def set_language(language):
    if language in LANGUAGES:
        # Clear any previous language setting first
        session.pop('language', None)
        
        # Set the new language in session
        session['language'] = language
        
        # Update in database if user is logged in
        if 'username' in session:
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    cursor = conn.cursor()
                    # Update user's language in database
                    cursor.execute("UPDATE users SET language=? WHERE name=?", 
                                  (language, session['username']))
                    conn.commit()
                    
                    # Verify the update worked
                    cursor.execute("SELECT language FROM users WHERE name=?", 
                                  (session['username'],))
                    result = cursor.fetchone()
                    if result and result[0] == language:
                        print(f"Successfully updated language to {language} in database")
                    else:
                        print(f"Failed to update language in database")
            except Exception as e:
                print(f"Error updating language in database: {str(e)}")
        
        # Force the session to be saved
        session.modified = True
        
        # Add debug info
        print(f"Language changed to {language} in session")
        print(f"Current session: {session}")
        
        # Flash a message to confirm
        flash(f"Language changed to {LANGUAGES[language]}", "success")
    
    # Use a cache-busting parameter to prevent browser caching
    return redirect(request.referrer + f"?lang_change={language}" if request.referrer else url_for('dashboard'))

@app.route('/debug_translation')
def debug_translation():
    current_lang = session.get('language', 'en')
    username = session.get('username', 'not logged in')
    
    # Check if user exists and get stored language
    db_lang = 'unknown'
    if username != 'not logged in':
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT language FROM users WHERE name=?", (username,))
                result = cursor.fetchone()
                if result:
                    db_lang = result[0]
        except Exception as e:
            db_lang = f"Error: {str(e)}"
    
    # Test translations for a few keys
    test_keys = ['welcome', 'login', 'register', 'dashboard', 'logout', 'settings']
    translations = {}
    for key in test_keys:
        translations[key] = get_text(key)
    
    # Get all available translations for the current language
    all_translations = {}
    for key in TRANSLATIONS.get(current_lang, {}):
        all_translations[key] = TRANSLATIONS[current_lang][key]
    
    return jsonify({
        'username': username,
        'session_language': current_lang,
        'db_language': db_lang,
        'sample_translations': translations,
        'available_translations': all_translations,
        'all_languages': LANGUAGES
    })

@app.route('/unit/<int:unit_id>', methods=['GET', 'POST'])
def unit(unit_id):
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    user_id = get_user_id(username)
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        if request.method == 'POST' and 'complete' in request.form:
            # First check if entry exists
            cursor.execute("SELECT id FROM progress WHERE user_id=? AND unit_number=?", (user_id, unit_id))
            if cursor.fetchone():
                cursor.execute("UPDATE progress SET completed=1 WHERE user_id=? AND unit_number=?", (user_id, unit_id))
            else:
                cursor.execute("INSERT INTO progress (user_id, unit_number, completed) VALUES (?, ?, 1)", (user_id, unit_id))
            
            conn.commit()
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST' and 'submit_project' in request.form:
            file = request.files.get('project_file')
            if file and allowed_file(file.filename):
                # Generate a unique filename to avoid conflicts
                original_filename = secure_filename(file.filename)
                filename = f"{unit_id}_{user_id}_{original_filename}"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save the file
                file.save(file_path)
                
                # Store only the filename in the database, not the full path
                cursor.execute("""
                    INSERT INTO submissions (user_id, unit_id, file_path)
                    VALUES (?, ?, ?)
                """, (user_id, unit_id, filename))
                
                # Check if progress record exists
                cursor.execute("SELECT id FROM progress WHERE user_id=? AND unit_number=?", (user_id, unit_id))
                if cursor.fetchone():
                    cursor.execute("""
                        UPDATE progress 
                        SET project_completed = 1 
                        WHERE user_id = ? AND unit_number = ?
                    """, (user_id, unit_id))
                else:
                    cursor.execute("""
                        INSERT INTO progress (user_id, unit_number, project_completed)
                        VALUES (?, ?, 1)
                    """, (user_id, unit_id))
                
                conn.commit()
                flash('Project submitted successfully!', 'success')
                return redirect(url_for('unit', unit_id=unit_id))
            
        cursor.execute("SELECT title, description, resources FROM projects WHERE unit_id=?", (unit_id,))
        project = cursor.fetchone()
        
        cursor.execute("SELECT title, content, file_path FROM materials WHERE unit_id=?", (unit_id,))
        materials = cursor.fetchall()
        
        cursor.execute("SELECT title, youtube_url, description FROM videos WHERE unit_id=?", (unit_id,))
        videos = cursor.fetchall()
        
        # Get vocabulary words for this unit
        cursor.execute("""
            SELECT id, word, definition, example, section 
            FROM words 
            WHERE unit_id=? 
            ORDER BY section, id
        """, (unit_id,))
        words = cursor.fetchall()
        
        progress = get_progress(user_id, unit_id)
        project_completed = progress[2] if progress else 0
        quiz_attempted = has_attempted_quiz(user_id, unit_id)
        
        # Get the quiz_id for this unit (added to fix the URL building error)
        quiz_id = None
        cursor.execute("SELECT id FROM quizzes WHERE unit_id=? LIMIT 1", (unit_id,))
        quiz_result = cursor.fetchone()
        if quiz_result:
            quiz_id = quiz_result[0]
    
    return render_template('unit.html',
                         username=username,
                         unit_id=unit_id,
                         project=project,
                         materials=materials,
                         videos=videos,
                         words=words,
                         project_completed=project_completed,
                         quiz_attempted=quiz_attempted,
                         quiz_id=quiz_id)  # Add quiz_id to template context

@app.route('/quiz/<int:unit_id>', methods=['GET', 'POST'])
def quiz(unit_id):
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    user_id = get_user_id(username)
    
    # Check if previous units are completed
    if unit_id > 1:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT completed FROM progress WHERE user_id=? AND unit_number=?", 
                          (user_id, unit_id-1))
            previous_unit = cursor.fetchone()
            
            if not previous_unit or previous_unit[0] != 1:
                flash('You need to complete the previous unit first!', 'warning')
                return redirect(url_for('unit', unit_id=unit_id-1))
    
    # Get motivational messages
    motivational_messages = [
        "You've got this! Every question is an opportunity to learn.",
        "Believe in yourself - you're capable of amazing things!",
        "Mistakes are proof you're trying. Keep going!",
        "Your effort today is your success tomorrow.",
        "Learning is a journey, not a destination. Enjoy the process!"
    ]
    
    motivation = random.choice(motivational_messages)
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Remove the quiz attempt check to allow multiple attempts
        
        if request.method == 'POST':
            try:
                cursor.execute("SELECT id, question, options, correct_answer FROM quizzes WHERE unit_id=?", (unit_id,))
                questions = cursor.fetchall()
                
                if not questions:
                    flash('No questions found for this quiz', 'error')
                    return redirect(url_for('unit', unit_id=unit_id))
                
                score = 0
                results = []
                for q in questions:
                    try:
                        q_id = q[0]
                        user_answer = request.form.get(f'q{q_id}')
                        correct = False
                        
                        if user_answer and int(user_answer) == q[3]:
                            score += 1
                            correct = True
                        
                        cursor.execute("SELECT explanation FROM quizzes WHERE id=?", (q_id,))
                        explanation = cursor.fetchone()[0]
                        
                        results.append({
                            'question': q[1],
                            'options': json.loads(q[2]),
                            'correct_index': q[3],
                            'user_answer': int(user_answer) if user_answer else None,
                            'explanation': explanation,
                            'correct': correct
                        })
                    
                    except Exception as e:
                        flash(f'Error processing question {q_id}: {str(e)}', 'error')
                        continue
                
                # Updated passing score to 3 out of 5
                passed = score >= 3
                
                if passed:
                    overall_result = f"Congratulations! You passed with {score}/{len(questions)} correct answers!"
                    success_messages = [
                        "Awesome job! You're making excellent progress!",
                        "You're crushing it! Keep up the fantastic work!",
                        "Success! Your hard work is paying off!",
                        "Brilliant! You're mastering this material!",
                        "Stellar performance! You should be proud!"
                    ]
                    motivation = random.choice(success_messages)
                else:
                    overall_result = f"You scored {score}/{len(questions)}. You need at least 3 correct answers to pass. Try again!"
                    retry_messages = [
                        "Don't worry, learning takes time. Let's try again!",
                        "So close! Review the feedback and give it another shot.",
                        "Every attempt brings you closer to mastery!",
                        "Keep going! Persistence is the key to success.",
                        "You've got this! Take what you've learned and try again."
                    ]
                    motivation = random.choice(retry_messages)
                
                # Only update progress if passed
                if passed:
                    # Check if progress record exists
                    cursor.execute("SELECT id FROM progress WHERE user_id=? AND unit_number=?", (user_id, unit_id))
                    if cursor.fetchone():
                        cursor.execute("""
                            UPDATE progress SET quiz_score=?, completed=1
                            WHERE user_id=? AND unit_number=?
                        """, (score, user_id, unit_id))
                    else:
                        cursor.execute("""
                            INSERT INTO progress (user_id, unit_number, quiz_score, completed) 
                            VALUES (?, ?, ?, 1)
                        """, (user_id, unit_id, score))
                
                # Always record the attempt
                cursor.execute("INSERT INTO quiz_attempts (user_id, unit_id, score) VALUES (?, ?, ?)", 
                             (user_id, unit_id, score))
                conn.commit()
                
                return render_template('quiz_result.html',
                                    username=username,
                                    unit_id=unit_id,
                                    score=score,
                                    total=len(questions),
                                    passed=passed,
                                    results=results,
                                    overall_result=overall_result,
                                    motivation=motivation)
                
            except Exception as e:
                flash(f'Error processing quiz: {str(e)}', 'error')
                return redirect(url_for('quiz', unit_id=unit_id))
        
        else:
            cursor.execute("SELECT id, question, options FROM quizzes WHERE unit_id=?", (unit_id,))
            questions = cursor.fetchall()
            
            if not questions:
                flash('No questions found for this quiz', 'error')
                return redirect(url_for('unit', unit_id=unit_id))
            
            question_list = []
            for row in questions:
                try:
                    options = json.loads(row[2])
                    question_list.append({
                        'id': row[0],
                        'question': row[1],
                        'options': options
                    })
                except json.JSONDecodeError:
                    flash('Error loading quiz options', 'error')
                    continue
            
            return render_template('quiz.html',
                                username=username,
                                unit_id=unit_id,
                                questions=question_list,
                                motivation=motivation)

@app.route('/ai_assistant', methods=['GET', 'POST'])
def ai_assistant():
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            try:
                if llm is None:
                    flash("AI assistant is not available at this time. Please try again later.", 'error')
                    return redirect(url_for('ai_assistant'))
                    
                result = llm.invoke(question)
                answer = result.content
                
                user_id = get_user_id(username)
                with sqlite3.connect(DB_NAME) as conn:
                    conn.execute("""
                        INSERT INTO qa_history (user_id, question, answer)
                        VALUES (?, ?, ?)
                    """, (user_id, question, answer))
                    conn.commit()
                
                return render_template('ai_assistant.html',
                                    show_answer=True,
                                    question=question,
                                    answer=answer)
            except Exception as e:
                flash(f"Error getting response: {str(e)}", 'danger')
                return redirect(url_for('ai_assistant'))
    
    return render_template('ai_assistant.html', show_answer=False)

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    if not session.get('authenticated'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        if llm is None:
            return jsonify({'error': 'AI assistant not available'}), 503
            
        result = llm.invoke(question)
        answer = result.content
        
        user_id = get_user_id(session['username'])
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                INSERT INTO qa_history (user_id, question, answer)
                VALUES (?, ?, ?)
            """, (user_id, question, answer))
            conn.commit()
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    user_id = get_user_id(username)
    
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        rating = request.form.get('rating')
        
        if feedback_text:
            try:
                # Ensure feedback table has correct structure
                check_and_fix_feedback_table()
                
                with sqlite3.connect(DB_NAME) as conn:
                    conn.execute('''
                        INSERT INTO feedback (user_id, feedback_text, rating)
                        VALUES (?, ?, ?)
                    ''', (user_id, feedback_text, rating))
                    conn.commit()
                flash('Thank you for your feedback!', 'success')
                return redirect(url_for('dashboard'))
            except sqlite3.Error as e:
                flash(f'Error submitting feedback: {str(e)}', 'danger')
    
    return render_template('feedback.html', username=username)

@app.route('/qa_history')
def qa_history():
    if not session.get('authenticated'):
        return redirect(url_for('password_gate'))
    
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    user_id = get_user_id(username)
    
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer, created_at 
            FROM qa_history 
            WHERE user_id=? 
            ORDER BY created_at DESC
        """, (user_id,))
        history = cursor.fetchall()
    
    return render_template('qa_history.html', 
                         username=username,
                         history=history)

# ---------- ADMIN ROUTES ----------
@app.route('/admin', methods=['GET'])
def admin_redirect():
    """Redirect to admin login page"""
    return redirect(url_for('admin_login'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    # Clear any existing admin sessions
    if request.method == 'GET':
        session.pop('admin', None)
        session.pop('admin_username', None)
    
    # Run check to make sure admin table is fixed    
    check_and_fix_admin_table()
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('admin/login.html')
            
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password FROM admin_users WHERE username=?", (username,))
            admin = cursor.fetchone()
            
            if admin and admin[1] and check_password_hash(admin[1], password):
                session['admin'] = True
                session['admin_username'] = username
                flash('Welcome to the admin panel', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials', 'danger')
                
    return render_template('admin/login.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats = get_admin_stats()
    return render_template('admin/dashboard.html', stats=stats)

@app.route('/admin/users')
@admin_required
def admin_users():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, language FROM users ORDER BY name")
        users = cursor.fetchall()
    
    return render_template('admin/users.html', users=users)

@app.route('/admin/submissions')
@admin_required
def admin_submissions():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, u.name as username, s.unit_id, s.file_path, s.submitted_at
            FROM submissions s
            JOIN users u ON s.user_id = u.id
            ORDER BY s.submitted_at DESC
        """)
        submissions = cursor.fetchall()
    
    return render_template('admin/submissions.html', submissions=submissions)

@app.route('/admin/download_submissions')
@admin_required
def admin_download_submissions():
    # Create a BytesIO object to store the ZIP file
    memory_file = io.BytesIO()
    
    # Create a ZIP file in memory
    with zipfile.ZipFile(memory_file, 'w') as zf:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.id, u.name as username, s.unit_id, s.file_path, s.submitted_at
                FROM submissions s
                JOIN users u ON s.user_id = u.id
                ORDER BY s.submitted_at DESC
            """)
            submissions = cursor.fetchall()
            
            # Add each submission file to the ZIP
            for submission in submissions:
                file_path = submission['file_path']
                if file_path:
                    # Look for file in uploads folder
                    real_path = os.path.join(UPLOAD_FOLDER, file_path)
                    
                    # Check if file exists
                    if os.path.exists(real_path):
                        # Create a name for the file in the ZIP
                        file_name = f"Unit{submission['unit_id']}_{submission['username']}_{file_path}"
                        zf.write(real_path, file_name)
    
    # Reset the file pointer to the beginning
    memory_file.seek(0)
    
    # Send the file for download
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'all_submissions_{timestamp}.zip'
    )

@app.route('/admin/update_user_language', methods=['POST'])
@admin_required
def admin_update_user_language():
    user_id = request.form.get('user_id')
    language = request.form.get('language')
    
    if user_id and language in LANGUAGES:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE users SET language = ? WHERE id = ?", (language, user_id))
                conn.commit()
                flash('User language updated successfully', 'success')
        except sqlite3.Error as e:
            flash(f'Error updating user language: {str(e)}', 'danger')
    else:
        flash('Invalid user or language selection', 'danger')
        
    return redirect(url_for('admin_users'))

@app.route('/admin/feedback')
@admin_required
def admin_feedback():
    # Ensure feedback table has correct structure
    check_and_fix_feedback_table()
    
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT feedback.id, users.name, feedback.feedback_text, 
                       feedback.rating, feedback.created_at 
                FROM feedback
                JOIN users ON feedback.user_id = users.id
                ORDER BY feedback.created_at DESC
            ''')
            feedback_items = cursor.fetchall()
        except sqlite3.Error:
            feedback_items = []
    
    return render_template('admin/feedback.html', feedback=feedback_items)

# Add the admin_add_word route with section support
@app.route('/admin/add_word', methods=['GET', 'POST'])
@admin_required
def admin_add_word():
    # Ensure words table exists
    check_and_create_words_table()
    
    if request.method == 'POST':
        try:
            unit_id = request.form['unit_id']
            word = request.form['word']
            definition = request.form['definition']
            example = request.form.get('example', '')  # This field is optional
            section = request.form.get('section', 1)   # Default to section 1 if not provided
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    INSERT INTO words (unit_id, word, definition, example, section)
                    VALUES (?, ?, ?, ?, ?)
                """, (unit_id, word, definition, example, section))
                conn.commit()
                flash('AI vocabulary word added successfully', 'success')
                return redirect(url_for('admin_add_word'))
        except Exception as e:
            flash(f'Error adding word: {str(e)}', 'danger')
    
    return render_template('admin/add_word.html')

@app.route('/admin/add_quiz', methods=['GET', 'POST'])
@admin_required
def admin_add_quiz():
    if request.method == 'POST':
        try:
            data = request.form
            unit_id = data['unit_id']
            question = data['question']
            options = [data[f'option{i}'] for i in range(1,4)]  # Updated to 5 options
            correct_answer = int(data['correct_answer'])
            explanation = data['explanation']
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    INSERT INTO quizzes (unit_id, question, options, correct_answer, explanation)
                    VALUES (?, ?, ?, ?, ?)
                """, (unit_id, question, json.dumps(options), correct_answer, explanation))
                conn.commit()
                flash('Quiz question added successfully', 'success')
                return redirect(url_for('admin_add_quiz'))
        except Exception as e:
            flash(f'Error adding quiz: {str(e)}', 'danger')
    
    return render_template('admin/add_quiz.html')

@app.route('/admin/add_material', methods=['GET', 'POST'])
@admin_required
def admin_add_material():
    if request.method == 'POST':
        try:
            unit_id = request.form['unit_id']
            file = request.files.get('file')
            
            # Check if file was provided
            if not file or file.filename == '':
                flash('Please select a file to upload', 'danger')
                return redirect(url_for('admin_add_material'))
                
            if allowed_file(file.filename):
                # Generate a secure filename
                original_filename = secure_filename(file.filename)
                filename = f"unit_{unit_id}_{original_filename}"
                
                # Save the file
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                
                # Auto-generate title from filename (remove extension)
                title = os.path.splitext(original_filename)[0].replace('_', ' ').replace('-', ' ').title()
                
                # Get any content if provided (now optional)
                content = request.form.get('content', '')
                
                with sqlite3.connect(DB_NAME) as conn:
                    conn.execute("""
                        INSERT INTO materials (unit_id, title, content, file_path)
                        VALUES (?, ?, ?, ?)
                    """, (unit_id, title, content, filename))
                    conn.commit()
                    flash('Material added successfully', 'success')
                    return redirect(url_for('admin_manage_content'))
            else:
                flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
        except Exception as e:
            flash(f'Error adding material: {str(e)}', 'danger')
    
    return render_template('admin/add_material.html')

@app.route('/admin/edit_material/<int:material_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_material(material_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM materials WHERE id=?", (material_id,))
        material = cursor.fetchone()
        
        if not material:
            flash('Material not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        
        # Convert to dict for easier access
        material = {
            'id': material[0],
            'unit_id': material[1],
            'title': material[2],
            'content': material[3],
            'file_path': material[4]
        }
    
    if request.method == 'POST':
        try:
            unit_id = request.form['unit_id']
            title = request.form['title']
            content = request.form.get('content', '')  # Now optional
            
            # Check if a new file was uploaded
            file = request.files.get('file')
            file_path = material['file_path']  # Default to existing file
            
            if file and file.filename:
                # A new file was uploaded
                if allowed_file(file.filename):
                    # Delete the old file if it exists
                    if material['file_path']:
                        old_file_path = os.path.join(UPLOAD_FOLDER, material['file_path'])
                        try:
                            os.remove(old_file_path)
                        except:
                            pass  # File might not exist, that's okay
                    
                    # Save the new file
                    original_filename = secure_filename(file.filename)
                    file_path = f"unit_{unit_id}_{original_filename}"
                    file.save(os.path.join(UPLOAD_FOLDER, file_path))
                else:
                    flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
                    return redirect(url_for('admin_edit_material', material_id=material_id))
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    UPDATE materials
                    SET unit_id=?, title=?, content=?, file_path=?
                    WHERE id=?
                """, (unit_id, title, content, file_path, material_id))
                conn.commit()
                flash('Material updated successfully', 'success')
                return redirect(url_for('admin_manage_content'))
                
        except Exception as e:
            flash(f'Error updating material: {str(e)}', 'danger')
    
    return render_template('admin/edit_material.html', material=material)

@app.route('/admin/add_video', methods=['GET', 'POST'])
@admin_required
def admin_add_video():
    if request.method == 'POST':
        try:
            title = request.form['title']
            youtube_url = request.form['youtube_url']
            description = request.form['description']
            unit_id = request.form['unit_id']
            
            # Extract YouTube video ID if full URL is provided
            if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                if 'v=' in youtube_url:
                    # Format: https://www.youtube.com/watch?v=VIDEO_ID
                    youtube_id = youtube_url.split('v=')[1].split('&')[0]
                elif 'youtu.be/' in youtube_url:
                    # Format: https://youtu.be/VIDEO_ID
                    youtube_id = youtube_url.split('youtu.be/')[1].split('?')[0]
                else:
                    youtube_id = youtube_url
            else:
                youtube_id = youtube_url  # Assume ID was provided directly
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    INSERT INTO videos (unit_id, title, youtube_url, description)
                    VALUES (?, ?, ?, ?)
                """, (unit_id, title, youtube_id, description))
                conn.commit()
                flash('Video added successfully', 'success')
                return redirect(url_for('admin_add_video'))
        except Exception as e:
            flash(f'Error adding video: {str(e)}', 'danger')
    
    return render_template('admin/add_video.html')

@app.route('/admin/add_project', methods=['GET', 'POST'])
@admin_required
def admin_add_project():
    if request.method == 'POST':
        try:
            title = request.form['title']
            description = request.form['description']
            resources = request.form['resources']
            unit_id = request.form['unit_id']
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    INSERT INTO projects (unit_id, title, description, resources)
                    VALUES (?, ?, ?, ?)
                """, (unit_id, title, description, resources))
                conn.commit()
                flash('Project added successfully', 'success')
                return redirect(url_for('admin_add_project'))
        except Exception as e:
            flash(f'Error adding project: {str(e)}', 'danger')
    
    return render_template('admin/add_project.html')

@app.route('/admin/manage_content')
@admin_required
def admin_manage_content():
    # Ensure words table exists
    check_and_create_words_table()
    
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all quizzes
        cursor.execute("SELECT id, unit_id, question FROM quizzes ORDER BY unit_id, id")
        quizzes = cursor.fetchall()
        
        # Get all materials
        cursor.execute("SELECT id, unit_id, title FROM materials ORDER BY unit_id, id")
        materials = cursor.fetchall()
        
        # Get all videos
        cursor.execute("SELECT id, unit_id, title FROM videos ORDER BY unit_id, id")
        videos = cursor.fetchall()
        
        # Get all projects
        cursor.execute("SELECT id, unit_id, title FROM projects ORDER BY unit_id, id")
        projects = cursor.fetchall()
        
        # Get all AI vocabulary words with section
        cursor.execute("SELECT id, unit_id, word, section FROM words ORDER BY unit_id, section, id")
        words = cursor.fetchall()
    
    return render_template('admin/manage_content.html', 
                          quizzes=quizzes,
                          materials=materials,
                          videos=videos,
                          projects=projects,
                          words=words)

@app.route('/admin/export_users', methods=['GET'])
@admin_required
def admin_export_users():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, language FROM users ORDER BY id")
            users = cursor.fetchall()
            
            headers = ['ID', 'Username', 'Language']
            csv_file = generate_csv_file(users, 'users.csv', headers)
            
            if csv_file:
                return send_file(csv_file, 
                               mimetype='text/csv',
                               as_attachment=True,
                               download_name='users.csv')
            else:
                flash('Error generating CSV file', 'danger')
                return redirect(url_for('admin_users'))
    except Exception as e:
        flash(f'Error exporting users: {str(e)}', 'danger')
        return redirect(url_for('admin_users'))

@app.route('/admin/export_progress', methods=['GET'])
@admin_required
def admin_export_progress():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.name, p.unit_number, p.completed, 
                       p.quiz_score, p.project_completed
                FROM users u
                LEFT JOIN progress p ON u.id = p.user_id
                ORDER BY u.name, p.unit_number
            """)
            progress = cursor.fetchall()
            
            # Convert to list format
            data = []
            for row in progress:
                data.append([
                    row['id'], row['name'], row['unit_number'], 
                    row['completed'], row['quiz_score'], row['project_completed']
                ])
            
            headers = ['User ID', 'Username', 'Unit', 'Completed', 'Quiz Score', 'Project Completed']
            csv_file = generate_csv_file(data, 'progress.csv', headers)
            
            if csv_file:
                return send_file(csv_file, 
                               mimetype='text/csv',
                               as_attachment=True,
                               download_name='user_progress.csv')
            else:
                flash('Error generating CSV file', 'danger')
                return redirect(url_for('admin_dashboard'))
    except Exception as e:
        flash(f'Error exporting progress: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/export_feedback', methods=['GET'])
@admin_required
def admin_export_feedback():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.name, f.feedback_text, f.rating, f.created_at
                FROM feedback f
                JOIN users u ON f.user_id = u.id
                ORDER BY f.created_at DESC
            """)
            feedback = cursor.fetchall()
            
            # Convert to list format
            data = []
            for row in feedback:
                data.append([
                    row['name'], row['feedback_text'], 
                    row['rating'], row['created_at']
                ])
            
            headers = ['Username', 'Feedback', 'Rating', 'Created At']
            csv_file = generate_csv_file(data, 'feedback.csv', headers)
            
            if csv_file:
                return send_file(csv_file, 
                               mimetype='text/csv',
                               as_attachment=True,
                               download_name='user_feedback.csv')
            else:
                flash('Error generating CSV file', 'danger')
                return redirect(url_for('admin_feedback'))
    except Exception as e:
        flash(f'Error exporting feedback: {str(e)}', 'danger')
        return redirect(url_for('admin_feedback'))

@app.route('/admin/reset_db', methods=['GET', 'POST'])
@admin_required
def admin_reset_db():
    if request.method == 'POST':
        confirmation = request.form.get('confirmation')
        if confirmation == 'RESET':
            if reset_database():
                # Re-initialize with sample data
                add_sample_data()
                flash('Database has been reset successfully', 'success')
            else:
                flash('Error resetting database', 'danger')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Incorrect confirmation text', 'danger')
    
    return render_template('admin/reset_db.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    session.pop('admin_username', None)
    flash('You have been logged out from admin panel', 'info')
    return redirect(url_for('admin_login'))

# ---------- ADMIN CONTENT MANAGEMENT ROUTES ----------

# Quiz management
@app.route('/admin/view_quiz/<int:quiz_id>')
@admin_required
def admin_view_quiz(quiz_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM quizzes WHERE id=?", (quiz_id,))
        quiz = cursor.fetchone()
        if not quiz:
            flash('Quiz not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        
        # Parse options from JSON
        try:
            options = json.loads(quiz['options'])
        except:
            options = []
            
        return render_template('admin/view_quiz.html', quiz=quiz, options=options)

@app.route('/admin/edit_quiz/<int:quiz_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_quiz(quiz_id):
    if request.method == 'POST':
        try:
            data = request.form
            unit_id = data['unit_id']
            question = data['question']
            options = [data[f'option{i}'] for i in range(1, 4)]
            correct_answer = int(data['correct_answer'])
            explanation = data['explanation']
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    UPDATE quizzes 
                    SET unit_id=?, question=?, options=?, correct_answer=?, explanation=?
                    WHERE id=?
                """, (unit_id, question, json.dumps(options), correct_answer, explanation, quiz_id))
                conn.commit()
                flash('Quiz updated successfully', 'success')
                return redirect(url_for('admin_manage_content'))
        except Exception as e:
            flash(f'Error updating quiz: {str(e)}', 'danger')
            
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM quizzes WHERE id=?", (quiz_id,))
        quiz = cursor.fetchone()
        if not quiz:
            flash('Quiz not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        
        # Parse options from JSON
        try:
            options = json.loads(quiz['options'])
        except:
            options = []
            
        return render_template('admin/edit_quiz.html', quiz=quiz, options=options)

@app.route('/admin/delete_quiz/<int:quiz_id>')
@admin_required
def admin_delete_quiz(quiz_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("DELETE FROM quizzes WHERE id=?", (quiz_id,))
            conn.commit()
            flash('Quiz deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting quiz: {str(e)}', 'danger')
    return redirect(url_for('admin_manage_content'))

# Material management
@app.route('/admin/view_material/<int:material_id>')
@admin_required
def admin_view_material(material_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM materials WHERE id=?", (material_id,))
        material = cursor.fetchone()
        if not material:
            flash('Material not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/view_material.html', material=material)

@app.route('/admin/delete_material/<int:material_id>')
@admin_required
def admin_delete_material(material_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM materials WHERE id=?", (material_id,))
            material = cursor.fetchone()
            
            # Delete the file if it exists
            if material and material['file_path']:
                file_path = os.path.join(UPLOAD_FOLDER, material['file_path'])
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            
            conn.execute("DELETE FROM materials WHERE id=?", (material_id,))
            conn.commit()
            flash('Material deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting material: {str(e)}', 'danger')
    return redirect(url_for('admin_manage_content'))

# Video management
@app.route('/admin/view_video/<int:video_id>')
@admin_required
def admin_view_video(video_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE id=?", (video_id,))
        video = cursor.fetchone()
        if not video:
            flash('Video not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/view_video.html', video=video)

@app.route('/admin/edit_video/<int:video_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_video(video_id):
    if request.method == 'POST':
        try:
            title = request.form['title']
            youtube_url = request.form['youtube_url']
            description = request.form['description']
            unit_id = request.form['unit_id']
            
            # Extract YouTube video ID if full URL is provided
            if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                if 'v=' in youtube_url:
                    # Format: https://www.youtube.com/watch?v=VIDEO_ID
                    youtube_id = youtube_url.split('v=')[1].split('&')[0]
                elif 'youtu.be/' in youtube_url:
                    # Format: https://youtu.be/VIDEO_ID
                    youtube_id = youtube_url.split('youtu.be/')[1].split('?')[0]
                else:
                    youtube_id = youtube_url
            else:
                youtube_id = youtube_url  # Assume ID was provided directly
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    UPDATE videos 
                    SET unit_id=?, title=?, youtube_url=?, description=?
                    WHERE id=?
                """, (unit_id, title, youtube_id, description, video_id))
                conn.commit()
                flash('Video updated successfully', 'success')
                return redirect(url_for('admin_manage_content'))
        except Exception as e:
            flash(f'Error updating video: {str(e)}', 'danger')
            
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE id=?", (video_id,))
        video = cursor.fetchone()
        if not video:
            flash('Video not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/edit_video.html', video=video)

@app.route('/admin/delete_video/<int:video_id>')
@admin_required
def admin_delete_video(video_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("DELETE FROM videos WHERE id=?", (video_id,))
            conn.commit()
            flash('Video deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting video: {str(e)}', 'danger')
    return redirect(url_for('admin_manage_content'))

# Project management
@app.route('/admin/view_project/<int:project_id>')
@admin_required
def admin_view_project(project_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        project = cursor.fetchone()
        if not project:
            flash('Project not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/view_project.html', project=project)

@app.route('/admin/edit_project/<int:project_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_project(project_id):
    if request.method == 'POST':
        try:
            title = request.form['title']
            description = request.form['description']
            resources = request.form['resources']
            unit_id = request.form['unit_id']
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    UPDATE projects 
                    SET unit_id=?, title=?, description=?, resources=?
                    WHERE id=?
                """, (unit_id, title, description, resources, project_id))
                conn.commit()
                flash('Project updated successfully', 'success')
                return redirect(url_for('admin_manage_content'))
        except Exception as e:
            flash(f'Error updating project: {str(e)}', 'danger')
            
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        project = cursor.fetchone()
        if not project:
            flash('Project not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/edit_project.html', project=project)

@app.route('/admin/delete_project/<int:project_id>')
@admin_required
def admin_delete_project(project_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            conn.commit()
            flash('Project deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting project: {str(e)}', 'danger')
    return redirect(url_for('admin_manage_content'))

# Word management
@app.route('/admin/view_word/<int:word_id>')
@admin_required
def admin_view_word(word_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM words WHERE id=?", (word_id,))
        word = cursor.fetchone()
        if not word:
            flash('Word not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/view_word.html', word=word)

@app.route('/admin/edit_word/<int:word_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_word(word_id):
    if request.method == 'POST':
        try:
            unit_id = request.form['unit_id']
            word = request.form['word']
            definition = request.form['definition']
            example = request.form.get('example', '')
            section = request.form.get('section', 1)
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    UPDATE words 
                    SET unit_id=?, word=?, definition=?, example=?, section=?
                    WHERE id=?
                """, (unit_id, word, definition, example, section, word_id))
                conn.commit()
                flash('Word updated successfully', 'success')
                return redirect(url_for('admin_manage_content'))
        except Exception as e:
            flash(f'Error updating word: {str(e)}', 'danger')
            
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM words WHERE id=?", (word_id,))
        word = cursor.fetchone()
        if not word:
            flash('Word not found', 'danger')
            return redirect(url_for('admin_manage_content'))
        return render_template('admin/edit_word.html', word=word)

@app.route('/admin/delete_word/<int:word_id>')
@admin_required
def admin_delete_word(word_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("DELETE FROM words WHERE id=?", (word_id,))
            conn.commit()
            flash('Word deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting word: {str(e)}', 'danger')
    return redirect(url_for('admin_manage_content'))

@app.route('/admin/fix_file_paths')
@admin_required
def fix_file_paths():
    results = []
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path FROM submissions WHERE file_path LIKE 'static/uploads%'")
        submissions = cursor.fetchall()
        
        for sub_id, file_path in submissions:
            # Extract just the filename without the path
            if '\\' in file_path:
                filename = file_path.split('\\')[-1]
            else:
                filename = file_path.split('/')[-1]
                
            results.append(f"ID {sub_id}: {file_path} → {filename}")
            
            # Update the database with the corrected path
            cursor.execute("UPDATE submissions SET file_path = ? WHERE id = ?", (filename, sub_id))
        
        conn.commit()
        
    return f"""
    <h2>Fixed {len(results)} file paths</h2>
    <pre>{'<br>'.join(results)}</pre>
    <a href="{url_for('admin_submissions')}" class="btn btn-primary">Return to Submissions</a>
    """

@app.route('/admin/view_submission/<int:submission_id>')
@admin_required
def view_submission(submission_id):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get submission details
            cursor.execute("""
                SELECT s.file_path, s.unit_id, s.user_id, u.name as username
                FROM submissions s
                JOIN users u ON s.user_id = u.id
                WHERE s.id = ?
            """, (submission_id,))
            
            submission = cursor.fetchone()
            
            if not submission or not submission['file_path']:
                flash("Submission file not found in database", "error")
                return redirect(url_for('admin_submissions'))
            
            # Clean up the file path - remove any path prefix if present
            file_path = submission['file_path']
            if '\\' in file_path:
                file_name = file_path.split('\\')[-1]
            elif '/' in file_path:
                file_name = file_path.split('/')[-1]
            else:
                file_name = file_path
                
            # Get full path to file
            full_path = os.path.join(UPLOAD_FOLDER, file_name)
            print(f"Attempting to download: {full_path}")
            
            if not os.path.exists(full_path):
                flash(f"File not found on server: {file_name}", "error")
                return redirect(url_for('admin_submissions'))
            
            # Create a descriptive filename for the download
            download_name = f"Unit{submission['unit_id']}_{submission['username']}_{file_name}"
            
            # Send the file with explicit parameters
            return send_file(
                full_path,
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=download_name
            )
                
    except Exception as e:
        print(f"Download error: {str(e)}")
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(url_for('admin_submissions'))

@app.route('/admin/debug_submission/<int:submission_id>')
@admin_required
def debug_submission(submission_id):
    debug_info = []
    
    try:
        debug_info.append(f"Checking submission ID: {submission_id}")
        
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get submission details
            cursor.execute("SELECT file_path, unit_id, user_id FROM submissions WHERE id=?", (submission_id,))
            submission = cursor.fetchone()
            
            if not submission:
                debug_info.append("ERROR: Submission not found in database")
                return f"<pre>{'<br>'.join(debug_info)}</pre>"
            
            debug_info.append(f"Found submission: {dict(submission)}")
            
            # Check file path
            if not submission['file_path']:
                debug_info.append("ERROR: Submission file_path is empty")
                return f"<pre>{'<br>'.join(debug_info)}</pre>"
            
            # Get user info
            cursor.execute("SELECT name FROM users WHERE id=?", (submission['user_id'],))
            user = cursor.fetchone()
            debug_info.append(f"User info: {dict(user) if user else 'Not found'}")
            
            # Clean up the file path if needed
            file_path = submission['file_path']
            if '\\' in file_path:
                clean_filename = file_path.split('\\')[-1]
            elif '/' in file_path:
                clean_filename = file_path.split('/')[-1]
            else:
                clean_filename = file_path
                
            # Full path to file
            full_path = os.path.join(UPLOAD_FOLDER, clean_filename)
            debug_info.append(f"Original file path: {file_path}")
            debug_info.append(f"Cleaned filename: {clean_filename}")
            debug_info.append(f"Full file path: {full_path}")
            
            # Check if file exists
            file_exists = os.path.exists(full_path)
            debug_info.append(f"File exists: {file_exists}")
            
            if not file_exists:
                debug_info.append(f"UPLOAD_FOLDER is configured as: {UPLOAD_FOLDER}")
                debug_info.append(f"UPLOAD_FOLDER absolute path: {os.path.abspath(UPLOAD_FOLDER)}")
                debug_info.append(f"Files in upload folder: {os.listdir(UPLOAD_FOLDER)}")
                return f"<pre>{'<br>'.join(debug_info)}</pre>"
            
            # File information
            file_size = os.path.getsize(full_path)
            debug_info.append(f"File size: {file_size} bytes")
            
            # Create direct download link
            direct_link = f"/admin/stream_file/{clean_filename}"
            debug_info.append(f"Direct download link: {direct_link}")
            
            html = f"""
            <pre>{'<br>'.join(debug_info)}</pre>
            <hr>
            <a href="{direct_link}" style="padding: 10px; background: blue; color: white; text-decoration: none;">
                Direct Download
            </a>
            <hr>
            <a href="{url_for('admin_submissions')}" style="padding: 10px; background: gray; color: white; text-decoration: none;">
                Back to Submissions
            </a>
            """
            return html
            
    except Exception as e:
        debug_info.append(f"ERROR: {str(e)}")
        return f"<pre>{'<br>'.join(debug_info)}</pre>"

@app.route('/admin/stream_file/<path:filename>')
@admin_required
def stream_file(filename):
    """Stream a file directly to the browser"""
    try:
        # Extract just the filename portion if it contains path elements
        if '\\' in filename:
            clean_filename = filename.split('\\')[-1]
        elif '/' in filename:
            clean_filename = filename.split('/')[-1]
        else:
            clean_filename = filename
            
        file_path = os.path.join(UPLOAD_FOLDER, clean_filename)
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", 404
            
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            # Stream the file in chunks
            def generate():
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    yield chunk
                    
        # Set appropriate headers
        headers = {
            'Content-Disposition': f'attachment; filename="{os.path.basename(clean_filename)}"',
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(file_size),
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        # Return a streaming response
        return app.response_class(
            generate(),
            headers=headers,
            direct_passthrough=True
        )
        
    except Exception as e:
        print(f"Stream error: {str(e)}")
        return f"Error: {str(e)}", 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    # Only initialize the database if needed, don't reset it automatically
    if not os.path.exists(DB_NAME):
        print("Database not found. Creating and initializing with sample data...")
        init_db()
        add_sample_data()
        print("Database initialized with sample data.")
    else:
        print("Using existing database.")
        check_and_create_words_table()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
