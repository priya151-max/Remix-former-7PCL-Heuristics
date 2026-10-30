"""Start the website:  python run.py   (env: HOST, PORT, DEVICE, OPENROUTER_API_KEY ...)"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("webapp.main:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")), workers=1, log_level="info")
