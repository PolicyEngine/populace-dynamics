#!/usr/bin/env bash
# Install Quarto CLI for static site builds (used by Vercel build container).
set -euo pipefail

QUARTO_VERSION="${QUARTO_VERSION:-1.6.40}"
INSTALL_DIR="${INSTALL_DIR:-.quarto-cli}"

curl -fsSL \
  "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.tar.gz" \
  -o quarto.tar.gz

mkdir -p "${INSTALL_DIR}"
tar -xzf quarto.tar.gz -C "${INSTALL_DIR}" --strip-components=1
rm quarto.tar.gz

"${INSTALL_DIR}/bin/quarto" --version
