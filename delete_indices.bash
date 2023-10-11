#!/bin/bash

# Elasticsearch server address
ELASTICSEARCH_URL="localhost:9200"

# Function to list all existing indices (excluding the first one)
list_indices() {
    curl -sX GET "$ELASTICSEARCH_URL/_cat/indices?v" | awk 'NR>1 {print $3}'
}

# Get the list of indices
INDICES=($(list_indices))

# Display existing indices
echo "Existing Indices:"
for INDEX in "${INDICES[@]}"; do
    echo "$INDEX"
done

# Loop through each index and send DELETE request
for INDEX in "${INDICES[@]}"; do
    # Delete index
    curl -sX DELETE "$ELASTICSEARCH_URL/$INDEX"
    echo "Deleted contents of index: $INDEX"
done
