from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from blackjack_game.app import app as game_app

main_app = FastAPI()

main_app.mount("/game", game_app)

@main_app.get("/")
async def root():
    return RedirectResponse(url="/game/")

if __name__ == "__main__":
    uvicorn.run(main_app, host="127.0.0.1", port=8080)
