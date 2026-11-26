.PHONY: help install pose features train test app clean all

help:
	@echo "Available commands:"
	@echo "  make install   - Install project dependencies from requirements.txt"
	@echo "  make pose      - Extract pose landmarks from video clips (YOLO-Pose)"
	@echo "  make features  - Compute biomechanical features and trajectory smoothing"
	@echo "  make train     - Train models and evaluate cross-validation benchmark"
	@echo "  make test      - Run automated pytest unit test suite"
	@echo "  make app       - Launch interactive Streamlit web dashboard"
	@echo "  make clean     - Remove Python cache artifacts (__pycache__, .pytest_cache)"
	@echo "  make all       - Run complete pipeline (features -> train -> test)"

install:
	pip install -r requirements.txt

pose:
	python src/pose.py

features:
	python src/features.py

train:
	python src/train.py

test:
	python -m pytest

app:
	streamlit run app/app.py

clean:
	@python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	@echo "Cleaned cache artifacts."

all: features train test
