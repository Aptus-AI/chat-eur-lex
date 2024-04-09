# Vectors Store Setup

To set up the vectors database, follow these steps:

Run the database through Docker Compose:
```sh
cd vector-store
docker-compose up -d
```

Index the documents inside the database:

```sh
# Build the Docker image for the database indexer.
docker build -t db-indexer .

# Run the indexer container with the necessary configurations.
docker run --rm --network qdrant_network -v /path/config/file:/app/config.yaml db-indexer
```

The path `/path/config/file.yaml` must contain a YAML file containing the parameters for the pre-processing and indexing of the laws.
