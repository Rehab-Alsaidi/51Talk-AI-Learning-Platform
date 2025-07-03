

````markdown
# 🌟 51Talk AI Learning Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)
![AI](https://img.shields.io/badge/AI-HuggingFace-orange.svg)
![Deploy](https://img.shields.io/badge/Deploy-Railway-purple.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

### 🚀 Revolutionary AI-Powered Learning Platform
*Empowering education through intelligent conversation and personalized learning experiences*

A comprehensive Flask-based learning management system with AI-powered assistance, multilingual support, and advanced quiz functionality. This platform provides interactive learning experiences with document-based Q&A, team management, progress tracking, and comprehensive administrative tools.

---

## 🎯 Overview

**51Talk AI Learning Platform** combines AI with education using HuggingFace language models to provide personalized, multilingual learning.

---

## 🌟 Features

<details>
<summary><strong>🎓 Advanced Learning Management</strong></summary>

- Structured interactive units
- AI vocabulary explanations
- Progress tracking
- Project submission with camp filters
- Multi-camp system: Middle East & Chinese
</details>

<details>
<summary><strong>🧠 AI-Powered Features</strong></summary>

- Document Q&A with HuggingFace
- Context-aware conversation memory
- Multilingual AI (English, 中文, العربية)
- Smart PDF/PowerPoint/text processing
</details>

<details>
<summary><strong>📝 Assessment System</strong></summary>

- One-time quizzes with review
- AI grading with pass/fail config
- Camp-based customized quizzes
</details>

<details>
<summary><strong>👥 Team Management</strong></summary>

- Team formation and scoring
- Real-time leaderboards
- Team performance analytics
</details>

<details>
<summary><strong>🔐 Enterprise Security</strong></summary>

- Email verification & password reset
- Role-based access
- API rate limiting
- Input validation
</details>

<details>
<summary><strong>📊 Admin Tools</strong></summary>

- Quiz/material upload
- User analytics
- System health & monitoring
</details>

<details>
<summary><strong>🚀 Production Ready</strong></summary>

- Railway/Docker deployment
- Gunicorn, PostgreSQL pooling
- Performance tracking
</details>

---

## 🛠️ Technology Stack

**Backend**: Flask, PostgreSQL, HuggingFace, LangChain, FAISS  
**Frontend**: Jinja2, Bootstrap 5, Vanilla JS  
**AI**: Llama-3-8B-Instruct, Sentence Transformers  
**Infrastructure**: Docker, Railway, Gunicorn, Health Checks

---

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 15+
- HuggingFace API Key  
Optional: Docker, Gmail (App password), Railway account

---

## 🚀 Quick Setup

### 🚀 Railway Deployment (Recommended)

```bash
git clone https://github.com/yourusername/51talk-ai-learning.git
cd 51talk-ai-learning
npm install -g @railway/cli
railway login
railway init
railway add postgresql
railway up
````

### 🧪 Docker Compose Setup

```bash
docker-compose up -d
docker-compose logs -f web
docker-compose ps
```

### 🧑‍💻 Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
createdb fiftyone_learning
psql -d fiftyone_learning -f init.sql
python app.py
```

---

## 📁 Project Structure (Partial)

```
51talk-ai-learning/
├── app.py
├── config.py
├── qa.py
├── init.sql
├── requirements.txt
├── docker-compose.yml
├── static/
│   └── uploads/
├── templates/
│   └── admin/
└── documents/
```

---

## 🔐 Security Highlights

* Password hashing (salted)
* Email verification
* CSRF/XSS/SQLi protection
* Session security
* Rate limiting

---

## 🧠 API Endpoints

| Method | Endpoint           | Description      |
| ------ | ------------------ | ---------------- |
| POST   | `/login`           | User login       |
| GET    | `/dashboard`       | User dashboard   |
| POST   | `/ask_ai_enhanced` | Ask AI assistant |
| GET    | `/admin/dashboard` | Admin dashboard  |

---

## 📈 Analytics

* Progress & quiz analytics
* Team leaderboard stats
* AI usage metrics
* System health metrics

---

