curl -X POST "http://holly-voice:9000/inference" \
    -F "file=@Holly.wav" \
    -F "temperature=0" \
    -F "response_format=text" > Holly.txt
