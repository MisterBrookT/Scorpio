from transformers import AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import evaluate, datasets
import numpy as np
from predictor.seq_predictor.model import prefill_predictor
from predictor.seq_predictor.config import PrefillPredictorConfig
from predictor.seq_predictor.model.prefill_predictor import prefill_predictor_model
import json
import torch
from argparse import ArgumentParser, Namespace
from predictor.seq_predictor.utils.model_loader import set_default_torch_dtype 
from scipy.stats import kendalltau
from predictor.seq_predictor.utils.file_utils import create_output_dirs, PathsContainer
import os
from tqdm import tqdm
import math
from predictor.seq_predictor.utils.logging import init_logger

def parse_args():
    parser = ArgumentParser("SeqPredictor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print-loss", action='store_true')
    parser.add_argument("--training-file", type=str, default="")
    parser.add_argument("--validation-file", type=str, default="")
    parser.add_argument("--test-file", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-name", type=str, default="mse")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wc", type=float, default=0.01)
    parser.add_argument("--loss", type=str, default='crossentropy')
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
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
    
    parser.add_argument("--check-exist", action='store_true', 
                        help="Check existing model")
    return parser.parse_args()

class SeqPredictorDataset(Dataset):
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
            
            logger.info(f"Balanced bin boundaries: {self.bin_boundaries}")

    def __len__(self):
        return len(self.data)

    def __len2classlabel__(self, length):
        if self.balanced_bins and self.bin_boundaries is not None:
            # Find which bin this length belongs to
            length = min(length, self.label_max_length)
            for i in range(len(self.bin_boundaries) - 1):
                if length >= self.bin_boundaries[i] and length < self.bin_boundaries[i+1]:
                    # Return label in reverse order (higher label for shorter sequences)
                    return self.num_bins - 1 - i
            return 0  # Default case (should not reach here if boundaries are correct)
        else:
            label = self.max_label - min(self.label_max_length-1, length) // self.label_group_size
            return label
        
    def __len2lengthlabel__(self, length):
        label = self.label_max_length  -  min(self.label_max_length, length) 
        return label
    
    def __classlabel2len__(self, label):
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
        class_label = self.__len2classlabel__(origin_len)
        length_label = self.__len2lengthlabel__(origin_len)

        return prompt, generated, origin_len, class_label, length_label


def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    
    # Initialize logger for terminal output only
    logger = init_logger()
    
    if args.check_exist:
        logger.info("Checking existing model...")
        model_path = os.path.join(args.job_dir, args.run_id, "finetuned")
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}. Exiting.")
            return
        else:
            logger.info(f"Model does not exist at {model_path}. Continuing...")

    logger.info("Loading tokenizer...")
    llama3_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefill_predictor_model_config = args.config # 'config_prefill_opt.txt'
    config = PrefillPredictorConfig.from_json(prefill_predictor_model_config)


    if config.model.num_labels == -1:
        config.model.num_labels = math.ceil(args.label_max_length / args.label_group_size)
    logger.info(f"num_labels: {config.model.num_labels}")


    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(pred_model=config.model.pred_model, num_labels=config.model.num_labels, mtype=config.model.mtype, activation=config.model.activation, max_length=config.model.max_length, max_batch_size=config.model.max_batch_size)
    predictor.model = predictor.model.to("cuda:0")

    # load data   
    training_dataset_path = args.training_file
    validation_dataset_path = args.validation_file
    test_dataset_path = args.test_file
    train_dataset = []
    validation_dataset = []
    test_dataset = []

    with open(training_dataset_path) as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            train_dataset.append(info)
    with open(validation_dataset_path) as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            validation_dataset.append(info)
    with open(test_dataset_path) as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            test_dataset.append(info)

    # load dataset
    train_dataset = SeqPredictorDataset(train_dataset, llama3_tokenizer, max_length=config.model.max_length, label_max_length=args.label_max_length, label_group_size=args.label_group_size, balanced_bins=args.balanced_bins, num_bins=args.num_bins)

    bin_boundaries = train_dataset.bin_boundaries
    validation_dataset = SeqPredictorDataset(validation_dataset, llama3_tokenizer, max_length=config.model.max_length, label_max_length=args.label_max_length, label_group_size=args.label_group_size, balanced_bins=args.balanced_bins, num_bins=args.num_bins, bin_boundaries=bin_boundaries)
    test_dataset = SeqPredictorDataset(test_dataset, llama3_tokenizer, max_length=config.model.max_length, label_max_length=args.label_max_length, label_group_size=args.label_group_size, balanced_bins=args.balanced_bins, num_bins=args.num_bins, bin_boundaries=bin_boundaries)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wc)
    optimizer.zero_grad()

    if args.loss == 'mse':
        loss_func = torch.nn.MSELoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()

    # Track the best off-by-2 accuracy and best model state
    best_model_state = None
    best_length_rmse = float('inf')

    for epoch in range(args.epoch):
        predictor.model.train()
        total_loss = 0
        idx = 0
        for prompt, _, origin_len, class_labels, _ in tqdm(train_dataloader):
            prompt = list(prompt)
            
            encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")
            
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
                
            with torch.autocast(device_type="cuda"):

                outputs = predictor(input_ids, attention_mask)
                
                class_labels = class_labels.reshape(1, -1)
                class_labels = class_labels.to("cuda")
                if args.loss == 'crossentropy':
                    assert class_labels.max().item() < predictor.model.num_labels
                    logits = outputs.view(-1, predictor.model.num_labels)
                    loss = loss_func(logits, class_labels.view(logits.size(0))) 
                else:
                    loss = loss_func(outputs.view(1, -1), class_labels) 
            
            if args.print_loss:
                logger.info(f"loss: {loss}") 
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
                        
            total_loss += loss.item()
            idx += 1
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        length_label_list = []
        class_label_list = []
        prediction_list = []
        prediction_len_list = []
        real_len_list = []
        predictor.model.eval()

        # validation
        with torch.no_grad():
            train_labels = []
            for prompt, _, origin_len, class_labels, length_labels in tqdm(validation_dataloader):
                prompt = list(prompt)

                encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs['input_ids'].to("cuda:0")
                attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
                with torch.autocast(device_type="cuda"):
                    outputs = predictor(input_ids, attention_mask)

                predicted_scores = outputs.argmax(dim=-1).tolist()
                
                length_label_list.extend(length_labels.tolist())
                class_label_list.extend(class_labels.tolist())
                prediction_list.extend(predicted_scores)
                prediction_len_list.extend([validation_dataset.__classlabel2len__(l) for l in predicted_scores])
                real_len_list.extend(origin_len.tolist())
    
            tau, score = kendalltau(length_label_list, prediction_list)
            logger.info("Kendall's Tau: %s, p-value: %s", tau, score)

            exact_match_acc = (np.array(class_label_list) == np.array(prediction_list)).sum() / len(class_label_list)
            logger.info("Exact match accuracy: %s", exact_match_acc)
            
            # Off-by-1 accuracy (predictions within 1 bin of the correct label)
            off_by_1_acc = (np.abs(np.array(class_label_list) - np.array(prediction_list)) <= 1).sum() / len(class_label_list)
            logger.info("Off-by-1 accuracy: %s", off_by_1_acc)
            
            # Off-by-2 accuracy (predictions within 2 bins of the correct label)
            off_by_2_acc = (np.abs(np.array(class_label_list) - np.array(prediction_list)) <= 2).sum() / len(class_label_list)
            logger.info("Off-by-2 accuracy: %s", off_by_2_acc)
            

            # Bin MSE (mean squared error between predicted and actual label bins)
            bin_mse = np.mean((np.array(class_label_list) - np.array(prediction_list))**2)
            logger.info("Bin MSE: %s", bin_mse)
            
            # Length MSE (mean squared error between predicted and actual sequence lengths)
            length_mse = np.mean((np.array(real_len_list) - np.array(prediction_len_list))**2)
            logger.info("Length MSE: %s", length_mse)   

            # Root Mean Squared Error (RMSE) between predicted and actual sequence lengths
            length_rmse = math.sqrt(length_mse)
            logger.info("Length RMSE: %s", length_rmse)

            # Check if current off-by-2 accuracy is better than the best seen so far
            if length_rmse < best_length_rmse:
                logger.info("New best length RMSE: %.4f (improved from %.4f)", length_rmse, best_length_rmse)
                best_length_rmse = length_rmse
                # Save the current model state
                best_model_state = {
                    'model_state_dict': predictor.model.state_dict(),
                    'test_metrics': {
                        'epoch': epoch,
                        'tau': tau,
                        'exact_match_acc': exact_match_acc,
                        'off_by_1_acc': off_by_1_acc,
                        'off_by_2_acc': off_by_2_acc,
                        'bin_mse': bin_mse,
                        'length_mse': length_mse,
                        'length_rmse': length_rmse
                    }
                }
            else:
                logger.info("Length RMSE did not improve. Current: %.4f, Best: %.4f", length_rmse, best_length_rmse)
                


    # test evaluation
    predictor.model.eval()
    length_label_list = []
    class_label_list = []
    prediction_list = []
    prediction_len_list = []
    real_len_list = []
    logger.info("Starting evaluation...")
    
    # If we have a best model from validation, load it for test evaluation
    if best_model_state is not None:
        logger.info("Loading best validation model for test evaluation (from epoch %d)", best_model_state['test_metrics']['epoch'] + 1)
        predictor.model.load_state_dict(best_model_state['model_state_dict'])
    
    with torch.no_grad():
        for prompt, _, origin_len, class_labels, length_labels in tqdm(test_dataloader):
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
            length_label_list.extend(length_labels.tolist())
            class_label_list.extend(class_labels.tolist())
            prediction_list.extend(predicted_scores)
            prediction_len_list.extend([test_dataset.__classlabel2len__(l) for l in predicted_scores])
            real_len_list.extend(origin_len.tolist())

    tau, score = kendalltau(length_label_list, prediction_list)
    logger.info("Kendall's Tau: %s, p-value: %s", tau, score)
    
    # exact acc
    exact_match_acc = (np.array(class_label_list) == np.array(prediction_list)).sum() / len(class_label_list)
    logger.info("Exact match accuracy: %s", exact_match_acc)
    
    # Off-by-1 accuracy
    off_by_1_acc = (np.abs(np.array(class_label_list) - np.array(prediction_list)) <= 1).sum() / len(class_label_list)
    logger.info("Off-by-1 accuracy: %s", off_by_1_acc)
    
    # Off-by-2 accuracy
    off_by_2_acc = (np.abs(np.array(class_label_list) - np.array(prediction_list)) <= 2).sum() / len(class_label_list)
    logger.info("Off-by-2 accuracy: %s", off_by_2_acc)

    # Bin MSE
    bin_mse = np.mean((np.array(class_label_list) - np.array(prediction_list))**2)
    logger.info("Bin MSE: %s", bin_mse)
    
    # Length MSE
    length_mse = np.mean((np.array(real_len_list) - np.array(prediction_len_list))**2)
    logger.info("Length MSE: %s", length_mse)

    # Root Mean Squared Error (RMSE) between predicted and actual sequence lengths
    length_rmse = math.sqrt(length_mse)
    logger.info("Length RMSE: %s", length_rmse)
            
    best_model_state['evaluate_metrics'] = {
        'tau': tau,
        'exact_match_acc': exact_match_acc,
        'off_by_1_acc': off_by_1_acc,
        'off_by_2_acc': off_by_2_acc,
        'bin_mse': bin_mse,
        'length_mse': length_mse,
        'length_rmse': length_rmse
    }

    paths = PathsContainer.from_args(args.job_dir, args.run_id, prefill_predictor_model_config)
    
    usage_config_path = os.path.join(paths.output_dir, "usage_config.json")
    
    finetuned_model_output_path = os.path.join(paths.output_dir, "finetuned")

    config.model.path = str(finetuned_model_output_path)

    create_output_dirs(paths.output_dir)
    
    PrefillPredictorConfig.to_json(config, usage_config_path)

    predictor.model.config.__dict__['num_labels'] = config.model.num_labels

    # If we found a better model based on validation performance, save it
    if best_model_state is not None:
        logger.info("Saving the best validation model from epoch %d", best_model_state['test_metrics']['epoch'] + 1)
        # Model is already loaded with best state from test evaluation
        
        # Save best metrics information alongside the model
        best_metrics_path = os.path.join(paths.output_dir, "best_metrics.json")
        with open(best_metrics_path, 'w') as f:
            json.dump({"evaluate_metrics": best_model_state['evaluate_metrics'], "test_metrics": best_model_state['test_metrics']}, f, indent=4)
    else:
        logger.warning("No model performed better than the initialization. Saving the final model anyway.")

    predictor.model = predictor.model.half()
    predictor.model.save_pretrained(finetuned_model_output_path)


if __name__ == "__main__":
    run()


