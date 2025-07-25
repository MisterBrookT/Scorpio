from fastapi import FastAPI, Request
import torch
import os
from predictor.seq_predictor.model.prefill_predictor import prefill_predictor_model
from predictor.seq_predictor.config import PrefillPredictorConfig
from predictor.seq_predictor.utils.model_loader import set_default_torch_dtype
from transformers import AutoTokenizer
import uvicorn
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser("Sequence Length Predictor Service")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the usage_config.json file in the model output directory")
    parser.add_argument("--label-max-length", type=int, default=8192)
    parser.add_argument("--label-group-size", type=int, default=82)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)

    return parser.parse_args()

app = FastAPI()

def label2len(label, label_max_length, label_group_size):
    # Calculate the max_label consistently
    if label_max_length % label_group_size == 0:
        max_label = label_max_length // label_group_size - 1
    else:
        max_label = label_max_length // label_group_size

    # Calculate the upper and lower bounds of token lengths for this label
    lower_bound = (max_label - label) * label_group_size
    upper_bound = (max_label - label + 1) * label_group_size
    return (lower_bound + upper_bound) / 2

@app.post("/predict_length")
async def predict_length(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    
    # Tokenize
    encoded_inputs = predictor.tokenizer(
        [prompt], 
        max_length=config.model.max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad(), torch.autocast(device_type='cuda'):
        outputs = predictor(input_ids, attention_mask)
    
    # Convert prediction to length
    predicted_label = outputs.argmax(dim=-1).item()
    predicted_length = label2len(predicted_label, LABEL_MAX_LENGTH, LABEL_GROUP_SIZE)
    
    return {"predicted_length": predicted_length}

if __name__ == "__main__":
    args = parse_args()

    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    print("device:", device)
    
    # Load model and config
    config = PrefillPredictorConfig.from_json(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pred_model)

    # Constants for length prediction
    LABEL_MAX_LENGTH = args.label_max_length
    LABEL_GROUP_SIZE = args.label_group_size

    # Initialize model
    with set_default_torch_dtype(torch.float32):
        predictor = prefill_predictor_model(
            pred_model=config.model.path,
            num_labels=config.model.num_labels,
            mtype=config.model.mtype,
            activation=config.model.activation,
            max_length=config.model.max_length,
            max_batch_size=config.model.max_batch_size,
            tokenizer_name=config.model.pred_model,
        )
    predictor.model = predictor.model.to(device)
    predictor.model.eval()
    
    print(f"Loaded model from {config.model.path}")
    print(f"Label max length: {LABEL_MAX_LENGTH}, Label group size: {LABEL_GROUP_SIZE}")
    print(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)