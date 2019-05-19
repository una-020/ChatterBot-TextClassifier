import glob


def parseData(corpus_path):
    # TODO
    pass


def parseLabel(corpus_path, field="age"):
    # TODO
    pass


if __name__ == "__main__":
    count = 0
    text = open("blogs/blogs.txt", "w+")
    gender_text = open("blogs/gender.txt", "w+")
    age_text = open("blogs/age.txt", "w+")
    profession_text = open("blogs/profession.txt", "w+")
    sunsign_text = open("blogs/sunsign.txt", "w+")

    for filename in glob.glob("blogs/*.xml"):
        try:
            file = open(filename, 'rb')
            # 3804152.female.16.Student.Sagittarius.xml
            categories = filename.split(".")
            for line in file.readlines():
                if line[:1].decode('UTF-8').isalpha():
                    line = line.decode("utf-8")
                    line = line.replace("\n", "")
                    line = line.replace("\r", "")
                    line = line.replace("\t", "")
                    text.write(line + "\n")
                    gender_text.write(categories[1] + "\n")
                    age_text.write(categories[2] + "\n")
                    profession_text.write(categories[3] + "\n")
                    sunsign_text.write(categories[4] + "\n")
        except Exception as e:
            print(e)

    text.close()
    gender_text.close()
    age_text.close()
    profession_text.close()
    sunsign_text.close()
