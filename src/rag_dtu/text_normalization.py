import json
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Constants
FIELDS_TO_SKIP = [
    "course title",
    "Language of instruction",
    "Point( ECTS )",
    "Danish title",
    "Registration Sign up",
    "Green challenge participation",
    "Last updated",
    "Engelsk titel"  # Added to skip Danish-specific content
]


def to_snake_case(s):
    """Convert a string to snake_case format."""
    s = s.strip().lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s


def clean_text(text):
    """Clean up spacing and punctuation in text."""
    text = re.sub(r'\s+([.,:%])', r'\1', text)  # remove space before punctuation
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()

def add_semester_info(entry):
    schedule = entry.get("Schedule", "").lower()

    semesters = []
    if "spring" in schedule:
        semesters.append("13 weeks spring")
    if "autumn" in schedule:
        semesters.append("13 weeks autumn")
    if any(month in schedule for month in ["january", "august", "july", "june"]):
        semesters.append("3 weeks january")
    entry["Semester"] = None
    entry["Semester"] = semesters if semesters else ["Unknown"]
    return entry


def attach_exam_dates(entry, course_code, exam_map, reexam_map):
    # Extract schedule and exam-related text
    schedule = entry.get("Schedule", "")
    examination = entry.get("Date of examination", "")

    # Extract schedule codes from both fields
    def extract_schedule_keys(text):
        matches = re.findall(r'\b[FE]\d[-]?[AB]\b', text.upper())
        return [key.replace("-", "") for key in matches]

    schedule_keys_sched = extract_schedule_keys(schedule)
    schedule_keys_exam = extract_schedule_keys(examination)

    # Merge the two
    if schedule_keys_sched == schedule_keys_exam:
        schedule_keys = schedule_keys_sched
    elif not schedule_keys_sched:
        schedule_keys = schedule_keys_exam
    elif not schedule_keys_exam:
        schedule_keys = schedule_keys_sched
    else:
        schedule_keys = list(dict.fromkeys(schedule_keys_sched + schedule_keys_exam))  # deduplicated

    # Detect months and append corresponding 3-week keys
    months = ['january', 'june', 'july', 'august']
    for month in months:
        if month in schedule.lower():
            schedule_keys.append(f"3-weeks {month}")
            break

    # Debug print
    print(f"Schedule keys: {schedule_keys}")

    # Try course code
    exam = exam_map.get(course_code)
    reexam = reexam_map.get(course_code)

    # Try schedule keys if needed
    if not exam:
        for sk in schedule_keys:
            if sk in exam_map:
                exam = exam_map[sk]
                print(f"Exam found via schedule key: {sk} → {exam}")
                break
    if not reexam:
        for sk in schedule_keys:
            if sk in reexam_map:
                reexam = reexam_map[sk]
                print(f"Reexam found via schedule key: {sk} → {reexam}")
                break

    # Update entry
    entry["Exam"] = exam
    entry["Reexam"] = reexam

    return entry



def normalize_course_entry(entry, exam_map, reexam_map):
    """Normalize course entry structure and enrich with semester and exam info."""
    import re

    normalized = {}

    # 1. Split course title
    if "course title" in entry:
        match = re.match(r"^(\d{5}|\w{2}\d{3})\b\s*(.*)$", entry["course title"])
        if match:
            normalized["course_code"] = match.group(1)
            normalized["course_name"] = match.group(2)
    
    # 2. Rename "Language of instruction" → "Language"
    if "Language of instruction" in entry:
        normalized["Language"] = entry["Language of instruction"].strip()
    
    # 3. Rename and convert "Point( ECTS )" → "ECTS"
    if "Point( ECTS )" in entry:
        try:
            normalized["ECTS"] = int(entry["Point( ECTS )"].strip())
        except ValueError:
            normalized["ECTS"] = entry["Point( ECTS )"].strip()

    # 4. Copy remaining fields (unless skipped)
    for key, value in entry.items():
        if key in FIELDS_TO_SKIP or key in ["course title", "Language of instruction", "Point( ECTS )"]:
            continue

        if isinstance(value, str):
            value = re.sub(r'\s+', ' ', value).strip()

        normalized[key.strip()] = value

    # 5. Add semester info
    normalized = add_semester_info(normalized)

    # 6. Attach exam and re-exam dates
    normalized = attach_exam_dates(normalized, normalized["course_code"], exam_map, reexam_map)

    return normalized



def normalize_text_fields(entry):
    """Normalize and clean text fields in a course entry."""
    normalized = {}

    for key, value in entry.items():
        new_key = to_snake_case(key)
        
        if isinstance(value, str):
            value = clean_text(value)

            # Make list from learning objectives
            if new_key == "learning_objectives":
                bullet_points = re.split(r'\.\s+(?=[A-Z])', value)  # split on period followed by capital
                value = "\n- " + "\n- ".join([v.strip(". ") for v in bullet_points if v.strip()])
            
            # Semi-structure course_type
            if new_key == "course_type":
                value = value.replace("\n", " ").replace("  ", " ")
                if "," in value:
                    parts = [v.strip() for v in value.split(",")]
                    value = "; ".join(parts)

        normalized[new_key] = value

    return normalized



def preprocess_course_text(course_dict, do_stemming=False, do_lemmatization=False):
    """Preprocess course text for NLP analysis."""
    # 1. Combine all string values into a single text
    full_text = " ".join(str(v) for v in course_dict.values() if isinstance(v, str))

    # 3. Lowercase
    text = full_text.lower()

    # 4. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    #tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # 7. Stemming / Lemmatization
    if do_stemming and not do_lemmatization:
        tokens = [stemmer.stem(word) for word in tokens]
    elif do_lemmatization and not do_stemming:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif do_stemming and do_lemmatization:
        # First lemmatization, then stemming
        tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]

    # 8. Rebuild the text
    processed_text = " ".join(tokens)

    return processed_text, len(tokens)


def build_course_data(course_dict, exam_by_code, reexam_by_code, do_stemming=False, do_lemmatization=False):
    """Build a processed course data object from raw course data."""
    # Step 1: Normalize structure and texts
    normalized = normalize_text_fields(normalize_course_entry(course_dict, exam_by_code, reexam_by_code))


    print(f"Normalized course data: {normalized}")
    # Step 2: Build text before preprocessing (combine string values)
    preprocessed_text = " ".join(str(v) for v in normalized.values() if isinstance(v, str))

    # Step 3: Complete preprocessing
    postprocessed_text, token_count = preprocess_course_text(normalized, do_stemming=do_stemming, do_lemmatization=do_lemmatization)

    # Step 4: Build final result
    result = {
        "course_code": normalized.get("course_code"),
        "preprocessed_course": preprocessed_text,
        "postprocessed_course": postprocessed_text,
        "metadata": {
            "course_code": normalized.get("course_code"),
            "course_name": normalized.get("course_name"),
            "schedule": normalized.get("schedule"),
            "exam": normalized.get("exam"),
            "reexam": normalized.get("reexam"),
            "semester": normalized.get("semester"),
            "ECTS": normalized.get("ects")
        }
    }

    return result, token_count

def normalize_query(query, do_stemming=False, do_lemmatization=False, remove_stopwords=False):
    """
    Normalize and preprocess a search query for consistent text matching.
    
    Args:
        query (str): The input search query to normalize
        do_stemming (bool): Whether to apply Porter stemming
        do_lemmatization (bool): Whether to apply WordNet lemmatization
        remove_stopwords (bool): Whether to remove English stopwords
    
    Returns:
        tuple: (normalized query string, list of tokens)
    """
    # 1. Lowercase
    text = query.lower()

    # 2. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Optional stopwords removal
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # 5. Stemming / Lemmatization
    if do_stemming and not do_lemmatization:
        tokens = [stemmer.stem(word) for word in tokens]
    elif do_lemmatization and not do_stemming:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif do_stemming and do_lemmatization:
        # First lemmatization, then stemming
        tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]

    # 6. Rebuild the text
    normalized_query = " ".join(tokens)

    return normalized_query, tokens

def main():
    """Main function to execute the script on all courses."""
    try:
        # Load JSON file with course entries
        with open("data/all_courses_info.json", "r", encoding="utf-8") as f:
            courses = json.load(f)

        with open("data/exam_schedule_dtu.json", "r") as f:
            exam_data = json.load(f)

        exam_by_code = exam_data["EXAM_BY_CODE"]
        reexam_by_code = exam_data["REEXAM_BY_CODE"]

        print(exam_by_code)
            
        processed_courses = {}
        total_tokens = 0

        print(f"Processing {len(courses)} courses...")
        
        # Process all courses
        for course_key, course_data in courses.items():
            # Skip courses with Danish content
            if "Engelsk titel" in course_data:
                continue
                
            processed, token_count = build_course_data(course_data, exam_by_code=exam_by_code, reexam_by_code=reexam_by_code, do_stemming=False, do_lemmatization=True)
            total_tokens += token_count
            processed_courses[course_key] = processed
        
        # Save the processed output to a new JSON file
        output_file = "data/processed_courses.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_courses, f, indent=2, ensure_ascii=False)

        print(f"✅ Processed {len(processed_courses)} courses and saved to '{output_file}'")
        print(f"Total tokens: {total_tokens}")
        
    except FileNotFoundError:
        print("❌ Error: Could not find input file 'all_courses_info.json'")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()