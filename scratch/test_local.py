import os
import sys
import time
import subprocess

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(base_dir, 'src', 'main.py')
    data_path = os.path.join(base_dir, 'data', 'candidates.jsonl')
    out_path = os.path.join(base_dir, 'output', 'test_submission.csv')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("==================================================")
    print(" RUNNING LOCAL END-TO-END PIPELINE TEST")
    print("==================================================")
    
    start = time.time()
    
    # Run the main pipeline as a subprocess
    result = subprocess.run([sys.executable, main_script, data_path, out_path])
    
    end = time.time()
    duration = end - start
    
    print("\n==================================================")
    if result.returncode == 0:
        print(f"OK: SUCCESS! Pipeline completed.")
        print(f" Runtime: {duration:.2f} seconds.")
        if duration > 300:
            print("WARNING: WARNING: Runtime exceeded 5 minutes (300s). You will be disqualified in the sandbox!")
        else:
            print("OK: Runtime is well within the 5-minute sandbox limit.")
            
        print(f" Output saved to: {out_path}")
    else:
        print(f"ERROR: FAILED! Pipeline crashed with exit code {result.returncode}.")
        print("Check the logs above for errors.")
    print("==================================================")

if __name__ == '__main__':
    main()
