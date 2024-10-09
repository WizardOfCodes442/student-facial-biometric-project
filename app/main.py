from fastapi import FastAPI
from controllers.controllers import router

# FastAPI app setup
app = FastAPI()

# Include routes from the router
app.include_router(router)