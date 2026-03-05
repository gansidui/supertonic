go build api_server_supertonic.go helper.go

pkill api_server_supertonic

export GIN_MODE=release

# for MacOS to set the onnxruntime lib path
# export ONNXRUNTIME_LIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib

nohup ./api_server_supertonic > /dev/null 2>&1 &

