import argparse
from .intake import parse_patient_message


def main() -> None:
    p = argparse.ArgumentParser(description="CareFlow M1 synthetic intake demo")
    p.add_argument("--model", required=True, help="LiteLLM model identifier")
    p.add_argument("--message", required=True, help="SYNTHETIC example only")
    args = p.parse_args()
    result = parse_patient_message(args.message, args.model)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
