To run python code in VS code please follow the steps in the link below
https://code.visualstudio.com/docs/python/python-tutorial#_create-a-virtual-environment

After running the file locally below endpoints will be accessible. Copy paste the url below in postman to populate the endpoint details.
curl http://127.0.0.1:8080/test
curl http://127.0.0.1:8080/get_model
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "small"}' http://127.0.0.1:8080/select_model
curl -X POST -F 'audio=@/path/to/your/audio/file.wav' http://127.0.0.1:8080/transcribe
