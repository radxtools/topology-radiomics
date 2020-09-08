python -m venv .env
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r .env-scripts/dev_requirements.txt
pip install -r requirements.txt