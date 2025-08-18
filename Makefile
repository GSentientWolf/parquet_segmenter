# Makefile helpers for signing scripts

SHELL := /bin/bash

.PHONY: sign-powershell
sign-powershell:
	@echo "Signing PowerShell script using pwsh wrapper"
	@if [ -z "$(PFX)" ]; then \
		echo "Please set PFX variable, e.g. make sign-powershell PFX=/path/to/cert.pfx SCRIPT=scripts/set_log_env.ps1"; \
		exit 2; \
	fi
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Please set SCRIPT variable, e.g. make sign-powershell SCRIPT=scripts/set_log_env.ps1 PFX=/path/to/cert.pfx"; \
		exit 2; \
	fi
	./scripts/sign_powershell.sh "$(SCRIPT)" "$(PFX)" "$(PFX_PWD)"
