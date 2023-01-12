.PHONY: lint
lint:
        python -m flake8 geomstats
        python -m black --check geomstats
        python -m bandit --exit-zero -r geomstats
        python -m mypy

.PHONY: autoformat
autoformat:
        python -m isort geomstats 
        python -m black geomstats

.PHONY: test
test:
        pytest -m geomstats
