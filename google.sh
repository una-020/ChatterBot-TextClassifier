#!/bin/bash
filename="GoogleNews-vectors-negative300.bin.gz"

file_id="0B7XkCwpI5KDYNlNUTTlSS21pQmM"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url
gzip -d GoogleNews-vectors-negative300.bin.gz
