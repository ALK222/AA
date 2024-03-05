#!/usr/bin/env bash

cd ./src/

rm -f ./requirements.txt

# iterate over each folder on src
for folder in */; do
    echo $folder
    pipreqs $folder --savepath $folder/requirements.txt
    cat $folder/requirements.txt >> requirements_aux.txt
    rm $folder/requirements.txt
done

sort requirements_aux.txt | uniq > requirements.txt
rm requirements_aux.txt