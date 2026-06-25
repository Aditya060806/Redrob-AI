import os
import datetime

class Logger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        
    def info(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] INFO: {msg}"
        print(formatted)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")
            
    def error(self, msg, exc_info=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] ERROR: {msg}"
        print(formatted)
        if exc_info:
            print(exc_info)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")
            if exc_info:
                f.write(str(exc_info) + "\n")
