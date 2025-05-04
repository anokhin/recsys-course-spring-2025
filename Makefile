.PHONY: all setup venv install run-sim run-dataclient clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all: setup run
run:
	cd botify && docker-compose down -v && docker-compose up -d --build --force-recreate --scale recommender=2

	echo "Waiting for botify to start..."
	sleep 10

	echo "Running sim..."
	cd sim && ../$(PYTHON) -m sim.run --episodes 1000 --config config/env.yml multi --processes 4
	# fixme add global paths

	cd script && ../$(PYTHON) dataclient.py --recommender 2 log2local ../data4exp/graphbased_vs_sticky_$(shell echo $$RANDOM | md5 | head -c 4)

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r sim/requirements.txt
	$(PIP) install -r botify/requirements.txt
	$(PIP) install paramiko scp

setup: install

run-sim: venv
	cd sim && ../$(PYTHON) -m sim.run --episodes 1000 --config config/env.yml multi --processes 4

run-dataclient: venv
	cd script && ../$(PYTHON) dataclient.py --recommender 2 log2local ../data4exp/graphbased_vs_sticky_$(shell echo $$RANDOM | md5 | head -c 4)

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
