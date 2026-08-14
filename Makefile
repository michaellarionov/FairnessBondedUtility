PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.mplconfig

.PHONY: test data figures clean

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -c "from fbu.data.adult import download; print(download())"

figures:
	$(PYTHON) examples/01_adult_baseline_curve.py --scorer logit
	$(PYTHON) examples/02_adult_full_comparison.py --scorer logit --n-runs 5 --seed-sweep
	$(PYTHON) examples/01_adult_baseline_curve.py --scorer lpm
	$(PYTHON) examples/02_adult_full_comparison.py --scorer lpm --n-runs 5

clean:
	rm -rf outputs .pytest_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
