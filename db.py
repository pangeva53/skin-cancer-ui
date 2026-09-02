import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

DB_NAME = "skin_lesion_app.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Ensure users table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # 2. Ensure predictions table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            patient_name TEXT DEFAULT 'Anonymous',
            timestamp TEXT NOT NULL,
            m1_diagnosis TEXT,
            m1_confidence REAL,
            m2_diagnosis TEXT,
            m2_confidence REAL,
            m3_diagnosis TEXT,
            m3_confidence REAL,
            image_data BLOB,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)

    # 3. Schema migration: Add missing columns if upgrading from the old table version
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "patient_name" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT DEFAULT 'Anonymous'")
    if "image_data" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN image_data BLOB")

    # 4. Default admin
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

def log_prediction(username, patient_name, m1_diag, m1_conf, m2_diag, m2_conf, m3_diag, m3_conf, image_bytes):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            username, patient_name, timestamp, 
            m1_diagnosis, m1_confidence, 
            m2_diagnosis, m2_confidence, 
            m3_diagnosis, m3_confidence,
            image_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        patient_name.strip(),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        m1_diag, round(m1_conf, 2),
        m2_diag, round(m2_conf, 2),
        m3_diag, round(m3_conf, 2),
        sqlite3.Binary(image_bytes)
    ))
    conn.commit()
    conn.close()

def get_user_history(username=None):
    conn = get_connection()
    if username and username != "admin":
        query = "SELECT id, patient_name, timestamp, m1_diagnosis, m1_confidence, m2_diagnosis, m2_confidence, m3_diagnosis, m3_confidence FROM predictions WHERE username = ? ORDER BY id DESC"
        df = pd.read_sql_query(query, conn, params=(username,))
    else:
        query = "SELECT id, username, patient_name, timestamp, m1_diagnosis, m1_confidence, m2_diagnosis, m2_confidence, m3_diagnosis, m3_confidence FROM predictions ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_image_by_id(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_data FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None