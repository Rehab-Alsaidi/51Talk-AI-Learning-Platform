CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    verification_code VARCHAR(64),
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS progress (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    unit_number INTEGER,
    completed INTEGER DEFAULT 0,
    quiz_score INTEGER DEFAULT 0,
    project_completed INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quizzes (
    id SERIAL PRIMARY KEY,
    unit_id INTEGER,
    question TEXT,
    options TEXT,
    correct_answer INTEGER,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    unit_id INTEGER,
    title TEXT,
    description TEXT,
    resources TEXT
);

CREATE TABLE IF NOT EXISTS materials (
    id SERIAL PRIMARY KEY,
    unit_id INTEGER,
    title TEXT,
    content TEXT,
    file_path TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    unit_id INTEGER,
    title TEXT,
    youtube_url TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS words (
    id SERIAL PRIMARY KEY,
    unit_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    one_sentence_version TEXT,
    daily_definition TEXT,
    life_metaphor TEXT,
    visual_explanation TEXT,
    core_elements TEXT,
    scenario_theater TEXT,
    misunderstandings TEXT,
    reality_connection TEXT,
    thinking_bubble TEXT,
    smiling_conclusion TEXT,
    section INTEGER DEFAULT 1
);


CREATE TABLE IF NOT EXISTS qa_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    unit_id INTEGER,
    score INTEGER,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
-- Create quiz_responses table to store individual question answers
CREATE TABLE IF NOT EXISTS quiz_responses (
    id SERIAL PRIMARY KEY,
    attempt_id INTEGER REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    question_id INTEGER REFERENCES quizzes(id) ON DELETE CASCADE,
    user_answer INTEGER,
    is_correct BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_quiz_responses_attempt_id ON quiz_responses(attempt_id);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_question_id ON quiz_responses(question_id);

-- Also add passed column to quiz_attempts table if it doesn't exist
ALTER TABLE quiz_attempts ADD COLUMN IF NOT EXISTS passed BOOLEAN DEFAULT FALSE;
CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    unit_id INTEGER,
    file_path TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    feedback_text TEXT,
    rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS admin_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
-- Team-related tables
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    team_lead_id INTEGER REFERENCES users(id),
    camp VARCHAR(50) NOT NULL, -- 'Middle East' or 'Chinese'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_members (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, user_id)
);

CREATE TABLE IF NOT EXISTS team_scores (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    score INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Insert default admin user (password: admin123)
INSERT INTO admin_users (username, password) 
VALUES ('admin', 'pbkdf2:sha256:150000$0dsQUPzZ$a785632d0a1f6a8c9b2cac47dfa153f39e9af92d05f38db2cbf82e4c9fbb4749')
ON CONFLICT (username) DO NOTHING;
