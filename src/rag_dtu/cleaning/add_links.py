
import json
import os


def add_program_link():
    with open('data/processed_programmes.json', 'r', encoding='utf-8') as file:
        program_data = json.load(file)

    with open('data/reference_data.json', 'r', encoding='utf-8') as file:
        ref_data = json.load(file)

    for name, url in zip(ref_data["study_programme_titles"], ref_data["study_programme_urls"]):
        name = name.replace(" ","_")
        if name in program_data:
            program_data[name]["metadata"]["url"] = url 
        else:
            print(name, " not found!!!!!!!!!!!!!!!!!!!!!!!!")

    with open('data/processed_programmes.json', 'w', encoding='utf-8') as output:
        json.dump(program_data, output, ensure_ascii=False, indent=4)



def add_course_link():
    with open('data/processed_courses.json', 'r', encoding='utf-8') as file:
        course_data = json.load(file)
    with open('data/reference_data.json', 'r', encoding='utf-8') as file:
        urls_data = json.load(file)

    for url in urls_data["course_urls"]: 
        course_numb = url[-5:]
        if course_numb in course_data:
            course_data[course_numb]["metadata"]["url"] = url
            
    with open('data/processed_courses.json', 'w', encoding='utf-8') as output:
        json.dump(course_data, output, ensure_ascii=False, indent=4)


def main():
    # add_course_link()
    add_program_link()


if __name__ == "__main__":
    main()
