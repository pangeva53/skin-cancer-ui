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
    
    # 1. Users table with role and status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL
        )
    """)
    
    # Schema check/migration for users table
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [row[1] for row in cursor.fetchall()]
    if "role" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    if "status" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")

    # 2. Predictions table
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
            image_data BLOB,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    # 3. Ensure default admin user exists
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    admin_user = cursor.fetchone()
    if not admin_user:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, status, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "admin", "active", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    else:
        # Guarantee admin has the admin role
        cursor.execute("UPDATE users SET role = 'admin', status = 'active' WHERE username = 'admin'")
        
    conn.commit()
    conn.close()

def register_user(username, password, role="user"):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), hash_password(password), role, "active", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return True, "Account registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, status FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        role, status = user
        if status != "active":
            return False, "This account is inactive. Please contact the administrator.", None
        return True, "Login successful", role
    return False, "Invalid username or password.", None

# --- Admin Management Queries ---
def get_all_users():
    conn = get_connection()
    query = "SELECT id, username, role, status, created_at FROM users ORDER BY id ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def toggle_user_status(user_id, current_status):
    conn = get_connection()
    cursor = conn.cursor()
    new_status = "inactive" if current_status == "active" else "active"
    cursor.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

# --- Prediction Logging & History Queries ---
def log_prediction(username, m1_diag, m1_conf, m2_diag, m2_conf, m3_diag, m3_conf, image_bytes):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            username, timestamp, 
            m1_diagnosis, m1_confidence, 
            m2_diagnosis, m2_confidence, 
            m3_diagnosis, m3_confidence,
            image_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        m1_diag, round(m1_conf, 2),
        m2_diag, round(m2_conf, 2),
        m3_diag, round(m3_conf, 2),
        sqlite3.Binary(image_bytes)
    ))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = get_connection()
    # Omitting 'id' from the displayed dataframe for users
    query = """
        SELECT timestamp, m1_diagnosis, m1_confidence, m2_diagnosis, m2_confidence, m3_diagnosis, m3_confidence 
        FROM predictions 
        WHERE username = ? 
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_user_records_with_images(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, image_data FROM predictions WHERE username = ? ORDER BY id DESC", (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows