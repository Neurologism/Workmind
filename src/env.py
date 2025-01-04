import dotenv
import os

dotenv.load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
FILES_DIRECTORY = os.getenv("FILES_DIRECTORY")
MODEL_DIRECTORY = os.getenv("MODEL_DIRECTORY")

if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set")

if not DB_NAME:
    raise EnvironmentError("DB_NAME is not set")

if not FILES_DIRECTORY:
    FILES_DIRECTORY = "./dataStorage"

if not MODEL_DIRECTORY:
    MODEL_DIRECTORY = FILES_DIRECTORY + "/models"
