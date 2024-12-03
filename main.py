from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from compare.ai import app as compare_app
from game.app import app as game_app

# 메인 FastAPI 앱 초기화
main_app = FastAPI()

# 서브 앱 마운트
main_app.mount("/game", game_app)
main_app.mount("/compare", compare_app)

# 루트 경로에서 기본 리디렉트
@main_app.get("/")
async def root():
    return RedirectResponse(url="/game/")

# FastAPI 앱 실행
if __name__ == "__main__":
    uvicorn.run(main_app, host="127.0.0.1", port=8080)
