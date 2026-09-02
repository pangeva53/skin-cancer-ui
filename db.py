import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "skin_lesion_app.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # 2. Prediction history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            m1_diagnosis TEXT,
            m1_confidence REAL,
            m2_diagnosis TEXT,
            m2_confidence REAL,
            m3_diagnosis TEXT,
            m3_confidence REAL,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    # Create default admin user if not existing (admin / admin123)
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            ("admin", hash_password("admin123"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username.strip(), hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return True, "Account registered successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    return user is not None

def log_prediction(username, m1_diag, m1_conf, m2_diag, m2_conf, m3_diag, m3_conf):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            username, timestamp, 
            m1_diagnosis, m1_confidence, 
            m2_diagnosis, m2_confidence, 
            m3_diagnosis, m3_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        m1_diag, round(m1_conf, 2),
        m2_diag, round(m2_conf, 2),
        m3_diag, round(m3_conf, 2)
    ))
    conn.commit()
    conn.close()