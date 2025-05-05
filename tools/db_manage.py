#!/usr/bin/env python3
"""
Database Management Tool

This script provides command-line utilities for managing the database,
including initialization, resetting, and adding sample data.
"""

import argparse
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import config
from backend.database.init_db import init_db, add_sample_data_to_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_manage")


def main():
    """Main entry point for the database management tool."""
    parser = argparse.ArgumentParser(description="Database management utilities")
    
    # Add command-line arguments
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize the database (create tables)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset the database (drop and recreate tables)'
    )
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Add sample data to the database'
    )
    parser.add_argument(
        '--connection-string',
        type=str,
        help='Database connection string (overrides environment variable)'
    )
    
    args = parser.parse_args()
    
    # Override connection string if provided
    if args.connection_string:
        os.environ['DATABASE_URL'] = args.connection_string
        logger.info(f"Using database connection string: {args.connection_string}")
    
    # Process commands
    if args.reset:
        logger.info("Resetting database...")
        config.reset_db()
        logger.info("Database reset complete")
        
        # If both reset and init are specified, init will be done by reset
        if args.init:
            args.init = False
    
    if args.init:
        logger.info("Initializing database...")
        config.init_db()
        logger.info("Database initialization complete")
    
    if args.sample_data:
        if not (args.init or args.reset):
            # If we haven't initialized or reset, make sure tables exist
            logger.info("Ensuring database tables exist...")
            config.init_db()
        
        logger.info("Adding sample data to database...")
        add_sample_data_to_db()
        logger.info("Sample data added successfully")
    
    # If no arguments were provided, show help
    if not (args.init or args.reset or args.sample_data):
        parser.print_help()


if __name__ == "__main__":
    main() 