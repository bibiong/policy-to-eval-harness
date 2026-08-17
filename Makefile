.PHONY: install demo report test lint clean all

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

demo:
	p2e run --config configs/models.yaml --out results/demo_sim
	p2e annotate --run results/demo_sim --simulate

report:
	p2e report --run results/demo_sim --out reports/findings_demo.md --charts reports/charts_demo

test:
	pytest

lint:
	ruff check src tests

all: demo report test

clean:
	rm -rf results/demo_sim reports/charts_demo/*.png reports/findings_demo.md
