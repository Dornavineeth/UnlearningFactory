.PHONY: quality style

check_dirs := scripts src tests #setup.py

quality:
	ruff check $(check_dirs)

style:
	ruff --format $(check_dirs)

test:
	CUDA_VISIBLE_DEVICES= pytest tests/
