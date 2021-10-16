kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data

python -m train

aws s3 mb s3://disaster-tweets-example-remote-storage
aws s3 cp binaries s3://disaster-tweets-example-remote-storage/binaries --recursive

rm -rf binaries/tokenizer
rm -rf binaries/model
