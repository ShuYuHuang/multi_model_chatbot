cd app
uvicorn whisper_server:app --host 10.100.100.106 --reload-dir ../reload --port 8001