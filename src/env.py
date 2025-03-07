import dotenv
import os

dotenv.load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set")

if not DB_NAME:
    raise EnvironmentError("DB_NAME is not set")
