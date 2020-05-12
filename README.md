# dhbw20-ml

Vorlesung Machine Learning an der DHBW 2020

Bitte vor jeder Vorlesung pullen, ist ein "Work in progress" ;-)

## Launch in Docker

First you have to select in `docker-compose.yml` whether you want to use Traefik or port forwarding.

```bash
docker-compose up -d
```

If you need to get a token to log into the web-interface run

```
docker logs <container_id>
```

(find the container_id by `docker ps -a`)

To stop:
```
docker-compose down
```
