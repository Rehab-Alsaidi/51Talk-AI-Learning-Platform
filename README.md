# 51Talk-AI-Learning-Platform

A comprehensive Flask-based learning platform with AI integration, designed for interactive educational experiences.

## Overview

The 51Talk AI Learning Platform is a web application that provides a structured learning environment integrated with AI assistance. It features a course-based learning structure, quizzes, project submissions, and AI-powered chat assistance for learners.

## Features

- **Multi-language Support**: English, Chinese, and Arabic interfaces
- **Interactive Learning**: Structured unit-based learning with materials, videos, and quizzes
- **Project Submissions**: Students can submit project files for assessment
- **AI Assistant**: Powered by Google's Gemini, providing real-time assistance
- **Progress Tracking**: Track completion of units, quiz scores, and project submissions
- **Admin Panel**: Comprehensive administration tools for content management
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rehab-Alsaidi/51talk-ai-learning.git
   cd 51talk-ai-learning
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root with the following:
   ```
   FLASK_SECRET_KEY=your_secret_key
   GEMINI_API_KEY=your_gemini_api_key
   ADMIN_PASSWORD=your_admin_password
   ```

5. Run the application:
   ```
   python app.py
   ```

## Configuration

### Environment Variables

- `FLASK_SECRET_KEY`: Secret key for session security
- `GEMINI_API_KEY`: Google Gemini API key for AI features
- `ADMIN_PASSWORD`: Default password for admin access

### Application Settings

You can modify these settings in `app.py`:

- `ACCESS_PASSWORD`: Gate password for initial access ("5151" by default)
- `UPLOAD_FOLDER`: Location for storing uploaded files
- `ALLOWED_EXTENSIONS`: File types allowed for upload
- `DB_NAME`: Database filename

## Usage

### User Access

1. Enter the access password (default: "5151") at the gateway
2. Enter a username to access the platform
3. Navigate through the learning units, complete quizzes, and submit projects
4. Use the AI assistant for help with course content

### Language Selection

Click on the language options in the navigation bar to switch between:
- English
- Chinese (中文)
- Arabic (العربية)

## Admin Panel

Access the admin panel at `/admin/login` with the default credentials:
- Username: `admin`
- Password: Default value from `ADMIN_PASSWORD` env variable

### Admin Features

- **Dashboard**: Overview of platform statistics
- **User Management**: View and manage user accounts and languages
- **Content Management**: Add and manage learning materials
  - Add quizzes
  - Upload learning materials
  - Add video links
  - Create project assignments
- **Submissions**: View and download student project submissions
- **Export Data**: Export users, progress, and feedback as CSV
- **System Management**: Reset database if needed

## File Structure

```
51talk-ai-learning/
├── app.py              # Main application file
├── database.db         # SQLite database
├── requirements.txt    # Dependencies
├── .env                # Environment variables (create this)
├── static/             # Static assets
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript files
│   └── uploads/        # User uploaded files
└── templates/          # HTML templates
    ├── admin/          # Admin panel templates
    └── ...             # Other templates
```

## Dependencies

- Flask: Web framework
- LangChain: AI integration framework
- Google Generative AI: For Gemini model
- SQLite: Database
- Werkzeug: Utilities for file handling and security
- Python-dotenv: Environment variable management

## Troubleshooting

### File Download Issues

If you encounter issues with file downloads in the admin panel:

1. Visit `/admin/fix_file_paths` to correct file path issues in the database
2. Ensure the `UPLOAD_FOLDER` directory exists and has proper permissions
3. Use the debug link in the admin submissions page to diagnose specific file issues

### AI Assistant Not Working

1. Verify your Gemini API key is correct in the `.env` file
2. Check console logs for API-related errors
3. Ensure internet connectivity for API calls

© 2025 51Talk AI Learning Platform. All rights reserved.
