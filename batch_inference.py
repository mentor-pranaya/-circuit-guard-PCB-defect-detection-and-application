# batch_inference.py
import argparse
from PIL import Image
import inference_backend as ib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="CSV with template,test columns")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default="outputs_module6")
    args = p.parse_args()

    # Load model
    model, device = ib.load_model(args.model, args.device)

    # Run batch
    summary = ib.batch_run_from_pairs(args.pairs, model, device, output_root=args.output_root)
    print(f"✅ Processed {len(summary)} pairs. See {args.output_root} for outputs.")

if __name__ == "__main__":
    main()
