#!/bin/bash

echo Starting docker services...
docker compose up -d

echo Opening web apps   # Using python3 instead of xdg-open, as python is more common among Linux machines
python3 -m webbrowser http://localhost:5252 || xdg-open http://localhost:5252
python3 -m webbrowser http://localhost:8999 || xdg-open http://localhost:8999

echo Platform has been activated.
