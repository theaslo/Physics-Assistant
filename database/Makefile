# Physics Assistant Database Management Makefile

# Load environment variables
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Default values
POSTGRES_HOST ?= localhost
POSTGRES_PORT ?= 5432
POSTGRES_DB ?= physics_assistant
POSTGRES_USER ?= physics_user

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help setup start stop status test migrate backup restore clean install

# Default target
all: help

help: ## Show this help message
	@echo "$(GREEN)Physics Assistant Database Management$(NC)"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment:"
	@echo "  Database: $(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)"
	@echo "  User: $(POSTGRES_USER)"

install: ## Install Python dependencies
	@echo "$(YELLOW)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

setup: ## Setup database environment (.env file)
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN).env file created. Please update with your settings.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

start: ## Start database services using Docker Compose
	@echo "$(YELLOW)Starting database services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Database services started$(NC)"
	@sleep 5
	@make status

stop: ## Stop database services
	@echo "$(YELLOW)Stopping database services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Database services stopped$(NC)"

status: ## Check database service status
	@echo "$(YELLOW)Checking database status...$(NC)"
	@docker-compose ps
	@echo ""
	@python3 scripts/test_connection.py

logs: ## View database logs
	@docker-compose logs -f postgres

migrate: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(NC)"
	python3 scripts/migrate.py migrate
	@echo "$(GREEN)Migrations completed$(NC)"

migrate-status: ## Show migration status
	@python3 scripts/migrate.py status

migrate-force: ## Force rerun all migrations
	@echo "$(YELLOW)Force running all migrations...$(NC)"
	python3 scripts/migrate.py migrate --force

reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		python3 scripts/migrate.py reset; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

test: ## Test database connections and basic operations
	@echo "$(YELLOW)Testing database connections...$(NC)"
	python3 scripts/test_connection.py

backup: ## Create full database backup
	@echo "$(YELLOW)Creating database backup...$(NC)"
	python3 scripts/backup.py backup --type full
	@echo "$(GREEN)Backup completed$(NC)"

backup-schema: ## Create schema-only backup
	@echo "$(YELLOW)Creating schema backup...$(NC)"
	python3 scripts/backup.py backup --type schema

backup-data: ## Create data-only backup
	@echo "$(YELLOW)Creating data backup...$(NC)"
	python3 scripts/backup.py backup --type data

restore: ## Restore database from backup (requires BACKUP_FILE parameter)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Error: BACKUP_FILE parameter required$(NC)"; \
		echo "Usage: make restore BACKUP_FILE=backup_file.sql.gz"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restoring database from $(BACKUP_FILE)...$(NC)"
	python3 scripts/backup.py restore --file "$(BACKUP_FILE)"

restore-drop: ## Restore database from backup with drop existing
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Error: BACKUP_FILE parameter required$(NC)"; \
		echo "Usage: make restore-drop BACKUP_FILE=backup_file.sql.gz"; \
		exit 1; \
	fi
	@echo "$(RED)WARNING: This will drop the existing database!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		python3 scripts/backup.py restore --file "$(BACKUP_FILE)" --drop; \
	else \
		echo "$(YELLOW)Restore cancelled$(NC)"; \
	fi

backup-list: ## List all available backups
	@python3 scripts/backup.py list

backup-status: ## Show backup status and statistics
	@python3 scripts/backup.py status

backup-cleanup: ## Remove old backups based on retention policy
	@echo "$(YELLOW)Cleaning up old backups...$(NC)"
	python3 scripts/backup.py cleanup

psql: ## Connect to database using psql
	@echo "$(YELLOW)Connecting to database...$(NC)"
	@PGPASSWORD=$(POSTGRES_PASSWORD) psql -h $(POSTGRES_HOST) -p $(POSTGRES_PORT) -U $(POSTGRES_USER) -d $(POSTGRES_DB)

adminer: ## Open Adminer web interface
	@echo "$(GREEN)Adminer web interface available at: http://localhost:8080$(NC)"
	@echo "Server: postgres"
	@echo "Username: $(POSTGRES_USER)"
	@echo "Password: [from .env file]"
	@echo "Database: $(POSTGRES_DB)"

clean: ## Clean up Docker containers and volumes
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Cleanup completed$(NC)"

dev-setup: setup start migrate ## Complete development setup
	@echo "$(GREEN)Development environment setup completed!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "1. Review the .env file and update settings if needed"
	@echo "2. Access Adminer at http://localhost:8080"
	@echo "3. Run 'make test' to verify everything works"

prod-setup: ## Production setup (without sample data)
	@echo "$(YELLOW)Setting up production database...$(NC)"
	@make setup
	@make start
	@echo "$(YELLOW)Running core schema only...$(NC)"
	python3 scripts/migrate.py migrate
	@echo "$(GREEN)Production database setup completed$(NC)"

# Development shortcuts
quick-reset: stop clean start migrate ## Quick reset for development
	@echo "$(GREEN)Development database reset completed$(NC)"

daily-backup: backup backup-cleanup ## Daily backup routine
	@echo "$(GREEN)Daily backup routine completed$(NC)">