sudo docker run \
-p 9200:9200 \
-e 'discovery.type=single-node' \
-e 'xpack.security.enabled=false' \
-e 'xpack.security.http.ssl.enabled=false' \
-e 'cluster.routing.allocation.disk.watermark.low=90%' \
-e 'cluster.routing.allocation.disk.watermark.high=98%' \
-e 'cluster.routing.allocation.disk.watermark.flood_stage=99%' \
docker.elastic.co/elasticsearch/elasticsearch:8.9.0