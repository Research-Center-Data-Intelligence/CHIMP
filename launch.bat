@echo off

echo Starting docker services...
docker compose up -d

echo Opening web apps
start "" http://localhost:5252
start "" http://localhost:8999

echo Platform has been activated. You can exit this window now.
pause
