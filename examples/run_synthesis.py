# An example script for running the test case synthesiser.

import json
from ragscope.synthesiser import synthesise_test_cases

# 1. Read the source document
with open("my_document.txt", "r") as f:
    document_text = f.read()

# 2. Call the synthesiser to generate test cases
# Make sure your 'ollama run llama3' server is running in another terminal!
test_cases = synthesise_test_cases(document_text)

# 3. Save the generated test cases to a JSON file
# We convert the Pydantic objects to dictionaries for JSON serialization
test_cases_as_dicts = [tc.model_dump() for tc in test_cases]

with open("test_cases.json", "w") as f:
    json.dump(test_cases_as_dicts, f, indent=4)

print("\nSuccessfully generated and saved test cases to 'test_cases.json'")