import sqlite3
from datetime import datetime


DB_PATH = "swings.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS swings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            club TEXT,
            video_path TEXT,
            address_frame_path TEXT,
            top_frame_path TEXT,
            impact_frame_path TEXT,
            finish_frame_path TEXT,
            tempo_ratio REAL,
            head_movement REAL,
            top_hand_height REAL,
            spine_maintenance REAL
        )
    """)
    
    conn.commit()
    conn.close()


def save_swing_to_db(timestamp, club, video_path, key_frame_paths, tempo, head_move, hand_height, spine_maint):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO swings (
            timestamp, club, video_path,
            address_frame_path, top_frame_path, impact_frame_path, finish_frame_path,
            tempo_ratio, head_movement, top_hand_height, spine_maintenance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, club, video_path,
        key_frame_paths.get("address"), key_frame_paths.get("top"),
        key_frame_paths.get("impact"), key_frame_paths.get("finish"),
        tempo, head_move, hand_height, spine_maint
    ))
    
    conn.commit()
    conn.close()

def get_last_n_swings(n=10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM swings
        ORDER BY id DESC
        LIMIT ?
    """, (n,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]