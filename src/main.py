from __future__ import annotations

from fastapi import FastAPI

from src.routers import api_router

app = FastAPI(title='Personas Core', version='0.1.0')
app.include_router(api_router)

if __name__ == '__main__':
	import uvicorn

	uvicorn.run('src.main:app', host='0.0.0.0', port=8001, reload=True)
