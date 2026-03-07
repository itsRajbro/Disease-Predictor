import subprocess
import sys

def on_starting(server):
    try:
        from app import init_db
        init_db()
        print("✅ DB initialized")
    except Exception as e:
        print(f"❌ DB init failed: {e}")