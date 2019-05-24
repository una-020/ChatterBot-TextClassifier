import sys
import argparse
import pickle as pkl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ipath', '--input_path', help='Input path',
                        required=True, type=str)
    parser.add_argument('-mpath', '--model_path', help='Model path',
                        required=True, type=str)
    args = parser.parse_args()

    model = pkl.load(open(args.model_path, "rb"))
    sentences = [line.strip() for line in open(args.input_path).readlines()]
    sentences = [line for line in sentences if len(line) > 0]

    testX = model.get_X(sentences, fit=False)
    y_pred = model.predict(testX)
    results = model.labeler.inverse_transform(y_pred)

    f = sys.stdout
    for res in results:
        f.write(res + "\n")


if __name__ == "__main__":
    main()
