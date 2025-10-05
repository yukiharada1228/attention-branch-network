uv run kaggle datasets download emmarex/plantdisease
mkdir -p data
mv plantdisease.zip data/plantdisease.zip
unzip data/plantdisease.zip -d data
rm data/plantdisease.zip