# Python ML Practice Projects

A collection of small supervised-learning exercises covering diabetes, house-price, iris, and Titanic classification problems. Each script is intentionally independent and designed for learning rather than production deployment.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

The house-price example accepts a portable dataset path instead of a machine-specific path:

```bash
python House_price-Predictions.py path/to/house_prices.csv
```

The CSV must include `area`, `bedrooms`, `bathrooms`, and `price` columns. The remaining scripts use their documented/sample datasets or scikit-learn data.
