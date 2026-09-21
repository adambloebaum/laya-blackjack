# Security and deployment scope

The application is a **local research tool**. It binds to `127.0.0.1` by default. It has no remote-user authentication and exposes model-loading and experiment controls, so do not expose its API directly to the internet.

Model releases contain Safetensors weights and data/configuration files. `blackjack fetch-model` pins a Hugging Face commit, checks the release inventory, and installs a complete directory atomically. It does not enable `trust_remote_code`. Optimizer recovery files are local training artifacts and are not distributed with the release.

Keep Hugging Face and GitHub credentials in their respective credential stores. Never commit tokens or paste them into issues, logs, or replay files.

Please report vulnerabilities through GitHub's private vulnerability reporting feature when available. Avoid posting working exploits or credentials in public issues.
