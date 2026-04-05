docker-start:
	docker run -d -p 8080:8080 jyxsu6/medagentbench:latest

docker-stop:
	docker ps -q --filter ancestor=jyxsu6/medagentbench:latest | xargs -r docker stop

# ── Experiments (L0–L4) ─────────────────────────────────────────────

paper-baseline:
	uv run python expt.py run --config-file conf/paper_baseline.yaml

structured-tools:
	uv run python expt.py run --config-file conf/structured_tools.yaml

pot:
	uv run python expt.py run --config-file conf/pot.yaml

codeact:
	uv run python expt.py run --config-file conf/codeact.yaml

orchestrate:
	uv run python expt.py run --config-file conf/orchestrate.yaml

# ── Quick tests (single case) ───────────────────────────────────────

quick-test:
	uv run python expt.py quick-test --config-file conf/paper_baseline.yaml dataset.n=1

quick-test-structured:
	uv run python expt.py quick-test --config-file conf/structured_tools.yaml dataset.n=1

quick-test-pot:
	uv run python expt.py quick-test --config-file conf/pot.yaml dataset.n=1

quick-test-codeact:
	uv run python expt.py quick-test --config-file conf/codeact.yaml dataset.n=1

# ── Results ──────────────────────────────────────────────────────────

list:
	uv run python -m secretagent.cli.results list results/*

average:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*
