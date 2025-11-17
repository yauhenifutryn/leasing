import shutil
import os
from dotenv import load_dotenv

load_dotenv()

def check_bin(name: str) -> None:
    path = shutil.which(name)
    status = "OK" if path else "NOT FOUND"
    print(f"{name}: {status} {f'({path})' if path else ''}")

def main() -> None:
    print("Checking system deps...")
    check_bin("ffmpeg")
    print("\nChecking API keys...")
    print("OPENAI_API_KEY:", "SET" if os.getenv("OPENAI_API_KEY") else "MISSING")
    print("HUGGINGFACE_TOKEN:", "SET" if os.getenv("HUGGINGFACE_TOKEN") else "MISSING (only needed for diarization)")
    print("\nDone.")

if __name__ == "__main__":
    main()
