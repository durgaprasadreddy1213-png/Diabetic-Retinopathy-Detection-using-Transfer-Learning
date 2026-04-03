import os
import uuid
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

APP_SECRET_KEY = "change-this-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.keras")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"jpg", "jpeg", "png"}

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------------------------------------------------------
# Load Model (Load Only Once)
# -----------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Make sure final_model.keras is inside model folder."
    )

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(path: str):
    img = load_img(path, target_size=(224, 224))  # MUST match training size
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_label(path: str) -> str:
    x = preprocess_image(path)

    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])

    return CLASS_NAMES[idx]

# -----------------------------------------------------------------------------
# Disable Browser Caching
# -----------------------------------------------------------------------------

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():

    prediction_result = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Please choose an image file.")
            return redirect(url_for("prediction_page"))

        if not allowed_file(file.filename):
            flash("Allowed file types: jpg, jpeg, png.")
            return redirect(url_for("prediction_page"))

        # 🔥 Always generate unique filename
        unique_filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        file.save(filepath)

        # 🔥 Predict fresh every time
        prediction_result = predict_label(filepath)

    return render_template("prediction.html", prediction=prediction_result)

# -----------------------------------------------------------------------------
# Run App
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # ⚠ VERY IMPORTANT: disable reloader
    app.run(debug=True, use_reloader=False)