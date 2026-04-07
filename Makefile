SHELL := /bin/bash

.PHONY: build build-prod up down logs

build:
	docker compose build dev

build-prod:
	docker build --target production -t aorta-seg:prod .

up:
	docker compose up -d dev

down:
	docker compose down

logs:
	docker compose logs -f dev
