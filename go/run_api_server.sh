go build api_server_supertonic.go helper.go

pkill api_server_supertonic

export GIN_MODE=release

nohup ./api_server_supertonic > /dev/null 2>&1 &

