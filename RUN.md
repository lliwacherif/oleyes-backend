# How to Run OLEYES

## Backend

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\activate

# 2. Navigate to the backend folder
cd backend

# 3. Start the FastAPI server (with hot-reload)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
