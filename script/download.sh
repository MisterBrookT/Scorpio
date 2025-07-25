
# download pre-generated dataset
mkdir -p dataset
huggingface-cli download Brookseeworld/dreamserve-dataset --local-dir dataset --repo-type dataset

# download pre-trained model
cd seq-predictor
mkdir -p model
huggingface-cli download Brookseeworld/dreamserve-predictor --local-dir model

