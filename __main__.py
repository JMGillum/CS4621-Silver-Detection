import argparse
from pathlib import Path
from __init__ import __version__


def SetupParser() -> argparse.ArgumentParser:
    """ Initializes all of the available command line arguments

    Returns: The ArgumentParser object for the arguments
        
    """
    version_parser = argparse.ArgumentParser(add_help=False)
    version_parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s v{__version__}"
    )
    verbose_parser = argparse.ArgumentParser(add_help=False)
    verbose_parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Turns on additional printing. Useful for debugging.",
    )
    input_dir_parser = argparse.ArgumentParser(add_help=False)
    input_dir_parser.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        help="Directory to load input files from",
    )
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="File to store the output",
    )
    output_dir_parser = argparse.ArgumentParser(add_help=False)
    output_dir_parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Directory to store the output files in",
    )

    parser = argparse.ArgumentParser(
        description="Classifies images as to whether they contain silver or not. The positional argument is required and specifies what operation to perform.",
        parents=[version_parser],
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "train",
        parents=[
            version_parser,
            verbose_parser,
            input_dir_parser,
            output_parser,
        ],
        help="Trains the model using the provided images."
    )
    subparsers.add_parser(
        "predict",
        parents=[
            version_parser,
            verbose_parser,
            input_dir_parser,
        ],
        help="Makes predictions on the provided images."
    )

    return parser

def Setup() -> dict:
    parser = SetupParser()
    args = vars(parser.parse_args())
    if args.get("verbose"):
        print(f"Arguments: {args}")
    if not args.get("command"):
        print("Invalid format. Use the `--help` flag for usage rules.",flush=True)
        exit(1)
    return args

def main():
    args = Setup()


    mode = args.get("command")
    if not mode:
        print("Invalid format. Use the `--help` flag for usage rules.",flush=True)
        exit(1)
    if mode == "train":
        import train
        train.Start2(root_image_path=args.get("input"),output_file=args.get("output"),verbose=args.get("verbose"))
    elif mode == "predict":
        import predict
        predict.Start(root_image_path=args.get("input"),verbose=args.get("verbose"))


if __name__ == "__main__":
    main()
