#!/bin/bash

# Directory paths
ARCHIVE_DIR=$1
RESULTS_DIR=$2

# Ensure the archive directory exists
if [ ! -d "$ARCHIVE_DIR" ]; then
  echo "Creating archive directory at $ARCHIVE_DIR"
  mkdir -p "$ARCHIVE_DIR"
fi

RETURN_FILES="$RESULTS_DIR/returns-viking-*.csv"
ARCHIVE_FILE="$ARCHIVE_DIR/returns-archive-$(date +%Y%m%d_%H%M%S).tar.gz"

# Check if there are return files to archive
if ls $RETURN_FILES 1> /dev/null 2>&1; then
  tar -czf "$ARCHIVE_FILE" -C "$RESULTS_DIR" $(basename -a $RETURN_FILES)
  echo "Archived return files to $ARCHIVE_FILE"
  
  # Remove the original return files after archiving
  rm -f $RETURN_FILES
else
  echo "No return files found to archive."
fi
