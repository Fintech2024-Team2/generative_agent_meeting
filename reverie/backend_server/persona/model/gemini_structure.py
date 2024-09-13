import json
import time
import google.generativeai as genai
from utils import *

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def _sleep(seconds=10):
  """Helper function to pause execution for a specified number of seconds.

  Args:
    seconds: The number of seconds to sleep. Defaults to 10.
  """
  time.sleep(seconds)

def _gemini_request(prompt):
  """Sends a request to the Gemini model and handles potential errors.

  Args:
    prompt: The prompt to send to the Gemini model.

  Returns:
    The text response from the Gemini model, or an empty string if an error occurs.
  """
  _sleep()
  try:
    response = model.generate_content(prompt)
    return response.text
  except Exception as e:
    print(f"Gemini Error: {e}")
    return ""

def generate_response(prompt, 
                       example_output,
                       special_instruction,
                       repeat=3,
                       fail_safe_response="Error: Unable to generate a valid response.",
                       validator=None,
                       cleanup=None,
                       verbose=False): 
  """Sends multiple requests to Gemini, validating and cleaning up the response.

  This function sends the given prompt to Gemini multiple times, applying validation
  and cleanup functions to the response. It returns a fail-safe response if all
  attempts fail.

  Args:
    prompt: The prompt to send to the Gemini model.
    example_output: An example of the expected output format.
    special_instruction: Specific instructions for formatting the output.
    repeat: The number of times to repeat the request. Defaults to 3.
    fail_safe_response: The response to return if all attempts fail. 
                         Defaults to a generic error message.
    validator: A function to validate the Gemini response. Defaults to None.
    cleanup: A function to clean up the Gemini response. Defaults to None.
    verbose: Whether to print debugging information. Defaults to False.

  Returns:
    The validated and cleaned up response from Gemini, or the fail_safe response 
    if all attempts fail.
  """

  prompt = f'"""\n{prompt}\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += f'Example output json:\n{{"output": "{example_output}"}}'

  if verbose: 
    print("Gemini Prompt:")
    print(prompt)

  for i in range(repeat):
    response = _gemini_request(prompt).strip()
    
    # Extract JSON output from response
    try:
      end_index = response.rfind('}') + 1
      response = response[:end_index]
      response = json.loads(response)["output"]
    except:
      if verbose:
        print(f"Attempt {i+1}: Unable to parse JSON from response.")
      continue

    # Validate and cleanup the response
    if validator and validator(response, prompt=prompt):
      return cleanup(response, prompt=prompt) if cleanup else response

    if verbose:
        print(f"Attempt {i+1}: Response did not pass validation: {response}")

  return fail_safe_response

def generate_prompt(input_values, prompt_template_file): 
  """Generates a prompt by replacing placeholders with input values.

  This function reads a prompt template from a file, replaces placeholders with 
  the given input values, and returns the resulting prompt.

  Args:
    input_values: The input value(s) to insert into the prompt. Can be a string
                  or a list of strings.
    prompt_template_file: The path to the prompt template file.

  Returns:
    The generated prompt string.
  """
  if isinstance(input_values, str):
    input_values = [input_values]
  input_values = [str(value) for value in input_values]

  with open(prompt_template_file, "r") as f:
    prompt = f.read()

  for count, value in enumerate(input_values):   
    prompt = prompt.replace(f"!<INPUT {count}>!", value)

  # Remove any comment block markers
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()

def get_embedding(text, model="models/text-embedding-004"):
  """Gets the text embedding for the given text using the specified model.

  Args:
    text: The text to embed.
    model: The name of the embedding model to use. 
           Defaults to "models/text-embedding-004".

  Returns:
    The text embedding from the Gemini model.
  """
  _sleep()
  text = text.replace("\n", " ").strip()
  if not text:
    text = "Empty input."
  return genai.embed_content(model=model, content=text, task_type='SEMANTIC_SIMILARITY')

if __name__ == '__main__':
  parameters = {"engine": "text-davinci-003", 
                "max_tokens": 50, 
                "temperature": 0, 
                "top_p": 1, 
                "stream": False,
                "frequency_penalty": 0, 
                "presence_penalty": 0, 
                "stop": ['"']}
  
  input_values = ["driving to a friend's house"]
  prompt_template_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(input_values, prompt_template_file)

  def _validate_response(gpt_response): 
    """Validates the GPT response to ensure it is a single word."""
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  
  def _cleanup_response(gpt_response):
    """Cleans up the GPT response by stripping whitespace."""
    return gpt_response.strip()

  output = generate_response(prompt, 
                             "driving",  # Example output
                             "The output should be a single word describing the activity.",
                             5,
                             validator=_validate_response,
                             cleanup=_cleanup_response,
                             verbose=True)

  print(output)