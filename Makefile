init:
	pip install -r requirements.txt

test:
	pytest

tests:
	pytest

.PHONY: init test
