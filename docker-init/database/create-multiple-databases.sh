#!/bin/bash
set -e
set -u

local echo_base="[Docker Postgres Init]"

function create_user_and_database() {
    local input=$1

    if [[ "$input" != *\[*\]@* ]]; then
        echo "$echo_base Error: Invalid input format '${input}'. Must be 'user[password]@database'."
        exit 1
    fi

    local user_and_password="${input%@*}"
    local database="${input#*@}"
    local user="${user_and_password%\[*\]}"
    local password="${user_and_password#*[}"
    password="${password%\]}"

    echo "$echo_base Creating database '$database' and user '$user' with password '$password'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname=postgres <<-EOSQL
        \set QUIET on
        CREATE USER $user WITH PASSWORD '$password';
        CREATE DATABASE $database;
        REVOKE CONNECT, TEMPORARY ON DATABASE $database FROM PUBLIC;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
        \set QUIET off
EOSQL
}

echo "$echo_base Revoking public privileges to postgres databases"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname=postgres <<-EOSQL
    \set QUIET on
    REVOKE CONNECT, TEMPORARY ON DATABASE postgres FROM PUBLIC;
    \set QUIET off
EOSQL

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "$echo_base Creating multiple databases: $(echo $POSTGRES_MULTIPLE_DATABASES | tr ' ' ', ')"
    
    for db in $POSTGRES_MULTIPLE_DATABASES; do
        create_user_and_database $db
    done
    
    echo "$echo_base Databases created."
fi
