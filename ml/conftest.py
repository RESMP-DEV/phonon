# ml/ runs on the GPU box in its own uv environments (see ml/AGENTS.md). The app's root
# `pytest` run must not collect it: its tests import jiwer, torch and NeMo, which the app CI
# does not install.
collect_ignore_glob = ["*"]
