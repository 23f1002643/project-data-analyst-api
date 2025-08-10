from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json

from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api")
async def analyze(request: Request):
    # Create a unique folder for this request
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    form = await request.form()
    question_text = None
    saved_files = {}

    # Save all uploaded files to the request folder
    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:  # It's a file
            file_path = os.path.join(request_folder, value.filename)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await value.read())
            saved_files[value.filename] = file_path  # Use filename as key, not field_name

            # If it's questions.txt, read its content
            if value.filename.lower() == "questions.txt":
                # Re-open the saved file to read content (since value.read() already consumed)
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    question_text = await f.read()
        else:
            saved_files[field_name] = value

    # Fallback: If no questions.txt, use the first uploaded file content as question_text
    if question_text is None and saved_files:
        # Find the first uploaded file path
        first_file_path = next(iter(saved_files.values()))
        # Read content of that file
        async with aiofiles.open(first_file_path, "r", encoding="utf-8") as f:
            question_text = await f.read()

    if question_text is None:
        return JSONResponse({"error": "No questions.txt file found and no fallback file available."}, status_code=400)

    # 4. Get code steps from LLM
    response = await parse_question_with_llm(
        question_text=question_text,
        uploaded_files=saved_files,
        folder=request_folder
    )

    print("LLM parse_question_with_llm response:", response)

    # 5. Execute generated code safely
    execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)

    print("run_python_code execution_result:", execution_result)

    count = 0
    while execution_result["code"] == 0 and count < 3:
        print(f"Error occurred while scraping, retry x{count}")
        new_question_text = f"{question_text}\nPrevious error: {execution_result['output']}"
        response = await parse_question_with_llm(
            question_text=new_question_text,
            uploaded_files=saved_files,
            folder=request_folder
        )
        print("Retry LLM response:", response)
        execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
        print("Retry execution_result:", execution_result)
        count += 1

    if execution_result["code"] != 1:
        return JSONResponse({"message": "Error occurred while scraping after retries."})

    # 6. Get answers from LLM
    gpt_ans = await answer_with_data(response["questions"], folder=request_folder)

    print("answer_with_data response:", gpt_ans)

    # 7. Execute answer code
    try:
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
    except Exception as e:
        print("Exception caught while executing answer code:", e)
        gpt_ans = await answer_with_data(response["questions"] + " Please follow the JSON structure.", folder=request_folder)
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)

    count = 0
    json_str_flag = True
    while final_result["code"] == 0 and count < 3:
        print(f"Error occurred while executing final code, retry x{count}")
        new_question_text = f"{response['questions']}\nPrevious error: {final_result['output']}"
        if not json_str_flag:
            new_question_text += " Follow the structure {'code': '', 'libraries': ''}"

        gpt_ans = await answer_with_data(new_question_text, folder=request_folder)
        print("Retry answer_with_data response:", gpt_ans)

        try:
            json_str_flag = False
            final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
            json_str_flag = True
        except Exception as e:
            print(f"Exception during retry execution: {e}")
            count -= 1  # Retry

        print("Retry final_result:", final_result)
        count += 1

    if final_result["code"] != 1:
        # If failed after retries, try to read result.json from folder if exists
        result_path = os.path.join(request_folder, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        else:
            return JSONResponse({"message": "Failed to generate valid result after retries."}, status_code=500)

    # If success
    result_path = os.path.join(request_folder, "result.json")
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse({"message": f"Error occurred while processing result.json: {e}"}, status_code=500)
