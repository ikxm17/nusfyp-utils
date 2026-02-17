#!/bin/bash
case "$1" in
    start)
        docker compose up -d && docker exec -it nerfstudio bash
        ;;
    rebuild)
        docker compose up -d --build && docker exec -it nerfstudio bash
        ;;
    stop)
        docker stop nerfstudio
        ;;
    *)
        echo "Usage: $0 {start|rebuild|stop}"
        exit 1
        ;;
esac