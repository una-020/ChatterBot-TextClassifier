import pandas as pd

def parseData(corpus_path):
    df = pd.read_csv(corpus_path)
    sentences = []
    for i in range(len(df)):
        sentences.append(df["text"][i])
    return sentences


def parseLabel(corpus_path):
    df = pd.read_csv(corpus_path)
    labels = []
    for i in range(len(df)):
        labels.append(df["sentiment"][i])
    return labels


def main():
    
    # # # Combining dev.tsv and train.tsv to get sentiment.tsv : NOTE : We still have unlabelled left.
    
    # # Reading files
    # file_path_train = "sentiment/train.csv"
    # file_path_dev = "sentiment/dev.csv"
    # df_train = pd.read_csv(file_path_train)
    # df_train.columns = ['sentiment', 'text']
    # df_dev = pd.read_csv(file_path_dev)
    # df_dev.columns = ['sentiment', 'text']
    
    # # Combine to sentiment_file
    # df_sentiment = pd.concat([df_train, df_dev])
      
    # # Sanity Check 
    # print ("Check: " + str(len(df_train)) + " + " + str(len(df_dev)) + " = " + str(len(df_sentiment)))
      
    # # Save final file 
    # df_sentiment.to_csv("sentiment/sentiment.csv", index = False)

    file_path = "data/sentiment/sentiment.csv"
    sents = parseData(file_path)
    labels = parseLabel(file_path)
    print(len(sents), len(labels))


if __name__ == "__main__":
    main()
