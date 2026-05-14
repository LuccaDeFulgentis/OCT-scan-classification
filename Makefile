train:
	python main.py

evaluate:
	python evaluate_only.py

install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	rm -f confusion_matrix.png