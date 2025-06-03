# main.py
from app.chat_engine import build_chat_ui


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch()
