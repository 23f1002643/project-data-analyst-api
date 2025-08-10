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
            saved_files[field_name] = file_path

            # If it's questions.txt, read its content
            if value.filename.lower() == "questions.txt":
                contents = await value.read()
                question_text = contents.decode("utf-8")
        else:
            saved_files[field_name] = value

    # Fallback: If no questions.txt, use the first file as question
    if question_text is None and saved_files:
        first_file_path = next(iter(saved_files.values()))
        async with aiofiles.open(first_file_path, "r") as f:
            question_text = await f.read()

    # ✅ 4. Get code steps from LLM
    response = await parse_question_with_llm(
        question_text=question_text,
        uploaded_files=saved_files,
        folder=request_folder
    )

    print(response)

    # ✅ 5. Execute generated code safely
    execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)

    print(execution_result)

    count = 0
    while execution_result["code"] == 0 and count < 3:
        print(f"Error occured while scrapping x{count}")
        new_question_text = str(question_text) + "previous time this error occured" + str(execution_result["output"])
        response = await parse_question_with_llm(
            question_text=new_question_text,
            uploaded_files=saved_files,
            folder=request_folder
        )

        print(response)

        execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)

        print(execution_result)

        count += 1

    if execution_result["code"] == 1:
        execution_result = execution_result["output"]
    else:
        return JSONResponse({"message": "error occured while scrapping."})

    # 6. get answers from llm
    gpt_ans = await answer_with_data(response["questions"], folder=request_folder)

    print(gpt_ans)

    # 7. Executing code
    try:
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
    except Exception as e:
        gpt_ans = await answer_with_data(response["questions"] + str("Please follow the json structure"), folder=request_folder)

        print("Trying after it caught under except block-wrong json format", gpt_ans)
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)

    count = 0
    json_str = 1
    while final_result["code"] == 0 and count < 3:
        print(f"Error occured while executing code x{count}")
        new_question_text = str(response["questions"]) + "previous time this error occured" + str(final_result["output"])
        if json_str == 0:
            new_question_text += "follow the structure {'code': '', 'libraries': ''}"

        gpt_ans = await answer_with_data(new_question_text, folder=request_folder)

        print(gpt_ans)

        try:
            json_str = 0
            final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
            json_str = 1
        except Exception as e:
            print(f"Exception occurred: {e}")
            count -= 1

        print(final_result)

        count += 1

    if final_result["code"] == 1:
        final_result = final_result["output"]
    else:
        result_path = os.path.join(request_folder, "result.json")
        with open(result_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    result_path = os.path.join(request_folder, "result.json")
    with open(result_path, "r") as f:
        try:
            data = json.load(f)
            return JSONResponse(content=data)
        except Exception as e:
            return JSONResponse({"message": f"Error occured while processing result.json: {e}"})
