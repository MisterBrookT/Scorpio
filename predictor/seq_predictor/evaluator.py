from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from predictor.seq_predictor.model import prefill_predictor
from predictor.seq_predictor.config import PrefillPredictorConfig
from predictor.seq_predictor.model.prefill_predictor import prefill_predictor_model
import json
import torch
from argparse import ArgumentParser
from predictor.seq_predictor.utils.model_loader import set_default_torch_dtype
from scipy.stats import kendalltau
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import math

def parse_args():
    parser = ArgumentParser("Sequence Length Predictor Evaluator")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the usage_config.json file in the model output directory")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to the test data file")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--label-max-length", type=int, default=8192, 
                                help="Maximum token length to consider (context length for Llama-3)")
    
    # Organize binning strategies into groups
    equal_width_group = parser.add_argument_group('Equal-width binning', 
                                                'Strategy that divides the token length range into equal-sized bins')
    equal_width_group.add_argument("--label-group-size", type=int, default=1,
                                  help="Size of each bin in tokens")
    
    equal_freq_group = parser.add_argument_group('Equal-frequency binning',
                                               'Strategy that creates bins with approximately equal number of samples')
    equal_freq_group.add_argument("--balanced-bins", action='store_true', 
                               help="Enable balanced binning (equal number of samples per bin)")
    equal_freq_group.add_argument("--num-bins", type=int, default=100, 
                                help="Number of bins to use when balanced-bins is enabled")
    
    return parser.parse_args()

class EvalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, label_max_length=8192, label_group_size=1,
                 balanced_bins=False, num_bins=100, bin_boundaries=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_max_length = label_max_length
        self.label_group_size = label_group_size
        self.balanced_bins = balanced_bins
        self.num_bins = num_bins if balanced_bins else label_max_length // label_group_size
        
        # For balanced bins, we'll store the bin boundaries
        self.bin_boundaries = bin_boundaries
        # For equal-width bins, we calculate the max label
        if label_max_length % label_group_size == 0:
            self.max_label = label_max_length // label_group_size - 1
        else:
            self.max_label = label_max_length // label_group_size
            
        # If balanced_bins is True but no bin_boundaries provided, calculate them
        if balanced_bins and bin_boundaries is None:
            # Calculate sequence lengths for all items
            lengths = []
            for item in tqdm(data, desc="Calculating sequence lengths"):
                length = len(tokenizer(item['generated'])['input_ids'])
                lengths.append(min(length, label_max_length))
            
            # Calculate bin boundaries based on percentiles
            self.bin_boundaries = []
            for i in range(num_bins + 1):
                percentile = 100 * i / num_bins
                # Calculate regular percentiles (ascending order)
                boundary = np.percentile(lengths, percentile)
                self.bin_boundaries.append(int(boundary))
            
            # Ensure the first boundary is 0
            self.bin_boundaries[0] = 0
            # Ensure the last boundary is at least label_max_length
            self.bin_boundaries[-1] = label_max_length
            
            # Sort boundaries just to be safe
            self.bin_boundaries = sorted(self.bin_boundaries)
            
            print(f"Balanced bin boundaries: {self.bin_boundaries}")

    def __len__(self):
        return len(self.data)

    def __len2label__(self, length):
        if self.balanced_bins and self.bin_boundaries is not None:
            # Find which bin this length belongs to
            length = min(length, self.label_max_length)
            for i in range(len(self.bin_boundaries) - 1):
                if length >= self.bin_boundaries[i] and length < self.bin_boundaries[i+1]:
                    # Return label in reverse order (higher label for shorter sequences)
                    return self.num_bins - 1 - i
            return 0  # Default case (should not reach here if boundaries are correct)
        else:
            # Updated implementation to match TrainingDataset
            label = self.max_label - min(self.label_max_length-1, length) // self.label_group_size
            return label
    
    def __label2len__(self, label):
        if self.balanced_bins and self.bin_boundaries is not None:
            # Convert label to bin index (reversed)
            bin_idx = self.num_bins - 1 - label
            if bin_idx < 0 or bin_idx >= len(self.bin_boundaries) - 1:
                # Handle out-of-range labels
                bin_idx = max(0, min(bin_idx, len(self.bin_boundaries) - 2))
            
            # Return the midpoint of the bin range
            lower_bound = self.bin_boundaries[bin_idx]
            upper_bound = self.bin_boundaries[bin_idx + 1]
            return (lower_bound + upper_bound) / 2
        else:
            # Original implementation
            lower_bound = (self.max_label - label) * self.label_group_size
            upper_bound = (self.max_label - label + 1) * self.label_group_size
            return (lower_bound + upper_bound) / 2

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        generated = item['generated']
        origin_len = len(self.tokenizer(item['generated'])['input_ids'])
        label = self.__len2label__(origin_len)

        return prompt, label, origin_len, generated

    @classmethod
    def from_csv(cls, csv_path, tokenizer, max_length=2048, label_max_length=8192, label_group_size=1):
        df = pd.read_csv(csv_path)
        data = []
        for _, row in df.iterrows():
            item = {
                'prompt': row['prompt'],
                'generated': row['generated'],  
            }
            data.append(item)
        print(f"from {csv_path}, load {len(data)} items")
        return cls(data, tokenizer, max_length, label_max_length, label_group_size)

    @classmethod
    def from_json(cls, json_path, tokenizer, max_length=2048, label_max_length=8192, label_group_size=1):
        dataset = []
        with open(json_path) as f:
            for jsonObj in f:
                info = json.loads(jsonObj)
                dataset.append(info)
        print(f"from {json_path}, load {len(dataset)} items")
        return cls(dataset, tokenizer, max_length, label_max_length, label_group_size)

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    prefill_predictor_model_config = args.config # 

    config = PrefillPredictorConfig.from_json(prefill_predictor_model_config)
    print(f"Loaded config from {args.config}")
    
    llama3_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Loaded tokenizer from {args.tokenizer}")
    

    with set_default_torch_dtype(torch.float32): 
        with torch.device('cuda'):
            predictor = prefill_predictor_model(
                pred_model=config.model.path,
                num_labels=config.model.num_labels,
                mtype=config.model.mtype,
                activation=config.model.activation,
                max_length=config.model.max_length,
                max_batch_size=config.model.max_batch_size,
                tokenizer_name=config.model.pred_model,
            )
    predictor.model = predictor.model.to("cuda:0")
        
    print(f"Loaded model from {config.model.path}")

    # load dataset
    dataset_path = args.file
    if dataset_path.endswith('.csv'):
        eval_dataset = EvalDataset.from_csv(
            dataset_path,
            llama3_tokenizer, 
            max_length=config.model.max_length, 
            label_max_length=args.label_max_length,
            label_group_size=args.label_group_size
        )
    else:
        eval_dataset = EvalDataset.from_json(
            dataset_path,
            llama3_tokenizer, 
            max_length=config.model.max_length, 
            label_max_length=args.label_max_length,
            label_group_size=args.label_group_size
        )
        
    # Convert to balanced binning if requested
    if args.balanced_bins:
        # We need to recreate the dataset with balanced bins
        data = eval_dataset.data  # Get the data from the original dataset
        eval_dataset = EvalDataset(
            data,
            llama3_tokenizer, 
            max_length=config.model.max_length, 
            label_max_length=args.label_max_length,
            label_group_size=args.label_group_size,
            balanced_bins=True,
            num_bins=args.num_bins
        )
        print(f"Using balanced binning with {args.num_bins} bins")

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # evaluate
    predictor.model.eval()
    true_labels = []
    predictions = []
    prediction_lens = []
    eval_data = []
    real_lens = []
    print("Starting evaluation...")
    with torch.no_grad():
        for prompt, labels, origin_len, generated in tqdm(eval_dataloader):
            prompt = list(prompt)

            encoded_inputs = predictor.tokenizer(
                prompt, 
                max_length=config.model.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
            
            with torch.autocast(device_type="cuda"):
                outputs = predictor(input_ids, attention_mask)

            predicted_scores = outputs.argmax(dim=-1).tolist()
            true_labels.extend(labels.tolist())
            predictions.extend(predicted_scores)
            prediction_lens.extend([eval_dataset.__label2len__(l) for l in predicted_scores])
            real_lens.extend(origin_len.tolist())
            
            # record data
            for i in range(len(prompt)):
                eval_data.append({
                    "prompt": prompt[i],
                    "generated": generated[i],
                    "origin_len": origin_len[i].item(),
                    "prediction_len": eval_dataset.__label2len__(predicted_scores[i]),
                    "label": labels[i].item(),
                    "prediction": predicted_scores[i]
                })
    
    # tau value
    tau, score = kendalltau(true_labels, predictions)
    print(f"Kendall's Tau: {tau}, p-value: {score}")
    
    # exact acc
    exact_match_acc = (np.array(true_labels) == np.array(predictions)).sum() / len(true_labels)
    print("Exact match accuracy: ", exact_match_acc)
    
    # Off-by-1 accuracy
    off_by_1_acc = (np.abs(np.array(true_labels) - np.array(predictions)) <= 1).sum() / len(true_labels)
    print("Off-by-1 accuracy: ", off_by_1_acc)
    
    # Off-by-2 accuracy
    off_by_2_acc = (np.abs(np.array(true_labels) - np.array(predictions)) <= 2).sum() / len(true_labels)
    print("Off-by-2 accuracy: ", off_by_2_acc)
    
    # Off-by-3 accuracy
    off_by_3_acc = (np.abs(np.array(true_labels) - np.array(predictions)) <= 3).sum() / len(true_labels)
    print("Off-by-3 accuracy: ", off_by_3_acc)

    # Bin MSE
    bin_mse = np.mean((np.array(true_labels) - np.array(predictions))**2)
    print("Bin MSE: ", bin_mse)
    
    # Bin RMSE
    bin_rmse = np.sqrt(bin_mse)
    print("Bin RMSE: ", bin_rmse)
    
    # Length MSE
    length_mse = np.mean((np.array(real_lens) - np.array(prediction_lens))**2)
    print("Length MSE: ", length_mse)
    
    # Length RMSE
    length_rmse = np.sqrt(length_mse)
    print("Length RMSE: ", length_rmse)

if __name__ == "__main__":
    main()