# Vectors Store Setup

Run database trough docker-compose:

`docker-compose up -d`

Index the documents inside the database:

```console
cd vector-store
docker build -t db-indexer .
docker run --rm --network qdrant_network -v /path/to/eurlex/data:/data/ -v /path/config/file:/app/config.yaml db-indexer
```

The path `/path/to/eurlex/data` must contain a list of txt files, where each file contains a parsed Eur-Lex document.
