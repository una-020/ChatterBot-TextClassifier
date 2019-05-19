def parseData(corpus_path):
    blog_list = []
    for blog in open(corpus_path):
        blog_list.append(blog)
    return blog_list


def parseLabel(corpus_path):
    label_list = []
    for label in open(corpus_path):
        label_list.append(label)
    return label_list


def main():
    file_path = "data/blog/blogs.txt"
    sents = parseData(file_path)
    labels = parseLabel("data/blog/gender.txt")
    print(len(sents), len(labels))
 

if __name__ == "__main__":
    main()
