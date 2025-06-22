"""
Configuration management for 51Talk AI Learning Platform.
Loads environment variables and provides app configuration settings.
"""

import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    # Flask configuration
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))
    DEBUG = False
    TESTING = False
    
    # File upload settings
    UPLOAD_FOLDER = 'static/uploads'
    DOCUMENTS_DIR = os.path.join(os.getcwd(), 'documents')
    VECTOR_DB_PATH = os.path.join(os.getcwd(), 'vector_db')
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'ppt', 'pptx', 'doc', 'docx'}
    
    # Access password for the application
    ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "5151")
    
    # Mail configuration
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
    
    # Database configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'fiftyone_learning')
    DB_USER = os.getenv('DB_USER', 'admin')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'admin123')
    
    # Create required directories
    @staticmethod
    def init_app(app):
        """Initialize application configuration."""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Use a separate test database
    DB_NAME = os.getenv('TEST_DB_NAME', 'fiftyone_testing')


class ProductionConfig(Config):
    """Production configuration."""
    # Production-specific settings can be added here
    pass


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}