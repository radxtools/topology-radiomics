rm -r .env
python3 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r .env-scripts/dev_requirements.txt
pip install -r requirements.txt