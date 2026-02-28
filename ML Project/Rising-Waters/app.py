import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ensure src is on path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from src.predict import predict_flood  # noqa: E402

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


def _prepare_payload(form_or_json):
    def _get(key, default=None, cast=float):
        val = form_or_json.get(key, default)
        if val in (None, ""):
            return default
        try:
            return cast(val)
        except Exception:
            return default

    state = form_or_json.get("state", "Unknown")
    year = _get("year", default=2000, cast=int)
    rainfall = _get("rainfall", default=0.0)
    prev_rain = _get("previous_year_rainfall", default=rainfall)
    temp = _get("temp", default=None)
    humidity = _get("humidity", default=None)
    cloud_cover = _get("cloud_cover", default=None)
    avgjune = _get("avgjune", default=None)
    sub = _get("sub", default=None)

    payload = {
        "state": state,
        "year": year,
        "annual_rainfall": rainfall,
        "monthly_rainfall_avg": rainfall / 12 if rainfall is not None else 0,
        "previous_year_rainfall": prev_rain,
        "rolling_mean_rainfall_3y": (rainfall + prev_rain) / 2 if prev_rain is not None else rainfall,
        "rainfall_dev_from_mean": 0.0,
        "Temp": temp if temp is not None else 0.0,
        "Humidity": humidity if humidity is not None else 0.0,
        "Cloud Cover": cloud_cover if cloud_cover is not None else 0.0,
        "avgjune": avgjune if avgjune is not None else 0.0,
        "sub": sub if sub is not None else 0.0,
    }
    return payload


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = _prepare_payload(request.form)
    try:
        risk, prob = predict_flood(payload)
        return render_template(
            "index.html",
            result={"risk": risk, "prob": f"{prob*100:.2f}%"},
            form_values=payload,
        )
    except Exception as exc:  # pragma: no cover - user-facing path
        return render_template("index.html", error=str(exc), form_values=payload)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    payload = _prepare_payload(data)
    try:
        risk, prob = predict_flood(payload)
        return jsonify({"risk": risk, "probability": prob})
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
