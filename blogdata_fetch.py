def get_blog_data(file="./blogs/blogs.txt"):
    blogs_list = []
    with open(file, "r") as blogs_file:
        for blog in blogs_file.readlines():
            blogs_list.append(blog)
    return blog
         

def get_gender(file="./blogs/gender.txt"):
    gender_list = []
    with open(file, "r") as gender_file:
        for gender in gender_file.readlines():
            gender_list.append(gender)
    return gender_list

def get_age(file="./blogs/age.txt"):
    age_list = []
    with open(file, "r") as age_file:
        for age in age_file.readlines():
            age_list.append(age)
    return age_list

def get_profession(file="./blogs/profession.txt"):
    prof_list = []
    with open(file, "r") as prof_file:
        for profession in prof_file.readlines():
            prof_list.append(profession)
    return prof_list

def get_sunsign(file="./blogs/sunsign.txt"):
    sunsign_list = []
    with open(file, "r") as sunsign_file:
        for sunsign in sunsign_file.readlines():
            sunsign_list.append(sunsign)
    return sunsign_list

#if __name__ == "__main__":
#    if sys.argv[-1] == "gender":
#        return get_gender()
#    elif sys.argv[-1] == "age":
#        return get_age()
#    elif sys.argv[-1] == "profession":
#        return get_profession() 
#    else:
#        return get_sunsign()
