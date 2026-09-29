.PHONY: cli run

ifeq (cli,$(firstword $(MAKECMDGOALS)))
  TOKENS ?= $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(TOKENS):;@:)
endif

cli:
	@if [ -n "$(TOKENS)" ]; then \
		cargo run --release -- --max-new-tokens $(TOKENS); \
	else \
		cargo run --release; \
	fi

run: cli
