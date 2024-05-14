import argparse

from e2yo.core import E2Yo


def _parse_args():
    parser = argparse.ArgumentParser(description="Changes all `e` to `ё` in txt file.")

    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="Path to input .txt file",
    )

    parser.add_argument(
        "--out_path",
        required=True,
        type=str,
        help="Path to output .txt file",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding of input and output files.",
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    e2yo = E2Yo()

    with open(args.input_path, encoding=args.encoding, mode="r") as file:
        input_text = file.read()

    output_text = e2yo.replace(input_text)

    with open(args.out_path, encoding=args.encoding, mode="w") as file:
        file.write(output_text)


if __name__ == "__main__":
    main()
