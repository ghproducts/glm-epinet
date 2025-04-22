# main.py
import argparse
import pickle
from pathlib import Path

from train import train
from inference import inference
from config import TrainArgs, ModelArgs

def main():
    parser = argparse.ArgumentParser(description="Train or run inference with NT + Epinet for HGT detection.")

    parser.add_argument("--input_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--out_name", type=str, default="output")
    parser.add_argument("--params_path", type=str)
    parser.add_argument("--epi_forwards", type=int, default=10)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--load_args", type=str, help="Path to .pkl file with saved TrainArgs")

    args = parser.parse_args()

    # Load TrainArgs from file or build manually
    if args.load_args:
        with open(args.load_args, 'rb') as f:
            config = pickle.load(f)
        print(f"Loaded TrainArgs from {args.load_args}")
    else:
        config = TrainArgs(
            input_path=args.input_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            out_name=args.out_name,
            params_path=args.params_path,
            epi_forwards=args.epi_forwards,
            num_classes=args.num_classes,
            model_args=ModelArgs(num_classes=args.num_classes)
        )

    if args.mode == "train":
        print("Starting training...")
        train(config)

        # Save the training configuration after training
        args_path = Path(config.output_path) / f"{config.out_name}_args.pkl"
        with open(args_path, 'wb') as f:
            pickle.dump(config, f)
        print(f"Saved training args to {args_path}")

    else:
        print("Running inference...")
        inference(config)

if __name__ == "__main__":
    main()
