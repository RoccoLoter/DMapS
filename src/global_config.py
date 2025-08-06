from pathlib import Path

repo_path = Path(__file__).absolute().parent.parent
tests_path = repo_path / "tests"
benchmark_path = tests_path / "benchmark"
hardware_info_path = tests_path / "hardware_info"

num_expr_rounds = 10

