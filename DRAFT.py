import asyncio
import importlib
import json
import os
import random
import re
from itertools import combinations
from typing import Any, Dict, Tuple, Union

import httpx
import numpy as np
import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pydantic import BaseModel


rapidapi_key = " "
openai_api_key = " "
openai_api_base = " "
api_version = "2024-02-15-preview"
proxy_url = "http://1.2.3.4:8080"


class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str


class APIClient:
    def __init__(self, timeout_seconds: int = 120, max_parallel: int = 200):
        self.timeout_seconds = timeout_seconds
        self.max_parallel = max_parallel
        self.llm_sem = asyncio.Semaphore(max_parallel)
        self.embedding_sem = asyncio.Semaphore(max_parallel)

    async def chat_completion(self, messages, temperature, top_p, max_tokens, model):
        headers = {
            "api-key": openai_api_key,
            "Content-Type": "application/json",
        }
        base_url = openai_api_base.rstrip("/")
        payload = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        async with self.llm_sem:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                verify=False,
                proxies=proxy_url,
            ) as client:
                response = await client.post(
                    f"{base_url}/chat/completions?api-version={api_version}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

    async def embedding(self, text, model="text-embedding-ada-002"):
        headers = {
            "api-key": openai_api_key,
            "Content-Type": "application/json",
        }
        base_url = openai_api_base.rstrip("/")
        payload = {"input": text, "model": model}

        async with self.embedding_sem:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                verify=False,
                proxies=proxy_url,
            ) as client:
                response = await client.post(
                    f"{base_url}/embeddings?api-version={api_version}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]


def validate_prompt_output(prompt_type: str, output: dict) -> bool:
    expected_schema = {
        "Explorer": {
            "User Query": str,
            "Parameters": dict,
        },
        "Analyzer": {
            "Suggestions for tool description": str,
        },
        "Rewriter": {
            "Rewritten description": str,
            "Suggestions for exploring": str,
        },
        "ToolDoc": {
            "tool_description": str,
        },
    }

    schema = expected_schema.get(prompt_type)
    if not schema or not isinstance(output, dict):
        return False

    for key, expected_type in schema.items():
        if key not in output:
            return False
        if not isinstance(output[key], expected_type):
            return False
    return True


async def openai_response(
    client: APIClient,
    messages,
    temperature,
    top_p,
    max_tokens,
    model,
    prompt_type: str,
    max_retries: int = 10,
):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            ans = await client.chat_completion(messages, temperature, top_p, max_tokens, model)
            cleaned_text = ans.strip("`json\n").strip("`\n")
            parsed = json.loads(cleaned_text)
            if validate_prompt_output(prompt_type, parsed):
                return parsed
            print(f"[{prompt_type}] invalid output format/type on attempt {attempt}: {parsed}")
        except Exception as e:
            last_exception = e
            print(f"[{prompt_type}] caught exception on attempt {attempt}: {type(e)} {e}")
            await asyncio.sleep(1)

    raise ValueError(
        f"{prompt_type} response validation failed after {max_retries} attempts. "
        f"last_exception={last_exception}"
    )


async def openai_embedding(client: APIClient, text):
    for attempt in range(1, 11):
        try:
            return await client.embedding(text)
        except Exception as e:
            print(f"[Embedding] caught exception on attempt {attempt}: {type(e)} {e}")
            await asyncio.sleep(1)
    raise ValueError("Embedding failed after 10 attempts")


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string



def prepare_tool_name_and_url(tools_root, info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")

    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name))
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from {tools_root}.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from {tools_root}.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string


def process_error(response):
    save_cache_flag = False
    switch_flag = False
    if "The request to the API has timed out. Please try again later, or if the issue persists" in str(response):
        return_dict = {"error": "API temporarily not working error...", "response": response}

    if "Your Client (working) ---> Gateway (working) ---> API (not working)" in str(response):
        return_dict = {"error": "API not working error...", "response": response}

    elif "Unauthorized" in str(response) or "unauthorized" in str(response):
        save_cache_flag = True
        return_dict = {"error": "Unauthorized error...", "response": response}

    elif "You are not subscribed to this API." in str(response):
        switch_flag = True
        return_dict = {"error": "Unsubscribed error...", "response": response}

    elif "Too many requests" in str(response):
        switch_flag = True
        return_dict = {"error": "Too many requests error...", "response": response}

    elif "You have exceeded" in str(response) or "you are being rate limited" in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Access restricted. Check credits balance or enter the correct API key." in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Oops, an error in the gateway has occurred." in str(response):
        switch_flag = True
        return_dict = {"error": "Gateway error...", "response": response}

    elif "Blocked User. Please contact your API provider." in str(response):
        switch_flag = True
        return_dict = {"error": "Blocked error...", "response": response}

    elif "error" in str(response):
        return_dict = {"error": "Message error...", "response": response}

    else:
        save_cache_flag = True
        return_dict = {"error": "", "response": response}
    return return_dict, save_cache_flag, switch_flag


def run(toolbench_code_string, toolbench_api_name, toolbench_input_params_str):
    success_flag = False
    switch_flag = False
    save_cache = False
    print(toolbench_code_string)
    try:
        exec(toolbench_code_string)
        eval_func_str = f"{toolbench_api_name}({toolbench_input_params_str})"
        new_func = eval(eval_func_str)
        response, save_cache, switch_flag = process_error(new_func)
        success_flag = True
    except Exception as e:
        response = {"error": f"Function executing {toolbench_code_string} error...\n{e}", "response": ""}
        save_cache = False
    return success_flag, switch_flag, response, save_cache



def dict_shorten(origin: dict, schema: dict):
    for key, value in list(origin.items()):
        if key not in schema:
            del origin[key]
        else:
            if isinstance(value, dict):
                dict_shorten(value, schema[key])
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        for item in value:
                            dict_shorten(item, schema[key][0])
    return origin


def observation_shorten(schema_root, response_dict, category, tool_name, api_name, strip_method):
    if strip_method == "filter" or (strip_method == "random" and random.random() > 0.5):
        if isinstance(response_dict["response"], dict):
            if os.path.exists(os.path.join(schema_root, category)):
                if os.path.exists(os.path.join(schema_root, category, tool_name + ".json")):
                    schema_dicts = json.load(open(os.path.join(schema_root, category, tool_name + ".json"), "r"))
                    api_list = schema_dicts["api_list"]
                    schema = None
                    for schema_dict in api_list:
                        schema_api_name = change_name(standardize(schema_dict["name"]))
                        if schema_api_name == api_name and len(schema_dict["schema"]) > 0:
                            schema = schema_dict["schema"]
                            break
                    if schema is not None:
                        response_dict["response"] = dict_shorten(response_dict["response"], schema)
    return str(response_dict["response"])


def get_rapidapi_response(
    input_dict: dict,
    api_customization: bool = False,
    tools_root: str = "data.toolenv.tools",
    schema_root: str = "data/toolenv/response_examples",
):
    # LLM이 생성한 파라미터 후보를 실제 API 호출로 연결하는 브리지.
    # DRAFT는 합성 데이터가 아닌 실제 상호작용 결과(응답/에러)로 문서를 학습적으로 개선한다.
    info = Info
    info.category = input_dict["category"]
    info.tool_name = input_dict["tool_name"]
    info.api_name = input_dict["api_name"]
    info.tool_input = input_dict["tool_input"]
    info.strip = input_dict["strip"]
    rapidapi_key = input_dict["rapidapi_key"]

    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(tools_root, info)
    tool_input = info.tool_input

    strip_method = info.strip
    if type(tool_input) == str:
        try:
            tool_input = json.loads(tool_input)
        except Exception:
            if tool_input == "":
                tool_input = {}
            else:
                print(f"Can not parse tool input into json: {tool_input}")
                response_dict = {"error": f"Tool input parse error...\n", "response": ""}
                return response_dict

    input_params_str = ""
    if len(tool_input) > 0:
        for key, value in tool_input.items():
            if isinstance(value, str):
                input_params_str += f'{key}="{value}", '
            else:
                input_params_str += f"{key}={value}, "
    if not api_customization:
        input_params_str += f"toolbench_rapidapi_key='{rapidapi_key}'"
    success_flag, switch_flag, response_dict, save_cache = run(code_string, api_name, input_params_str)
    observation = observation_shorten(
        schema_root,
        response_dict,
        standard_category,
        tool_name.replace(f"_for_{standard_category}", ""),
        api_name,
        strip_method,
    )
    result = str(observation)[:2048]
    return {"error": response_dict["error"], "response": result}


async def compute_similarity_and_bleu(client: APIClient, reference_sentence, candidate_sentence):
    # 적응형 종료를 위한 하이브리드 유사도:
    # 의미 유사도(임베딩 코사인) + 표면 문자열 유사도(BLEU).
    reference_sentence_embedding, candidate_sentence_embedding = await asyncio.gather(
        openai_embedding(client, reference_sentence),
        openai_embedding(client, candidate_sentence),
    )
    similarity = cosine_similarity(reference_sentence_embedding, candidate_sentence_embedding)

    reference = [reference_sentence.lower().split()]
    candidate = candidate_sentence.lower().split()

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    delta = (bleu_score + similarity) / 2
    return delta


async def process_api_info(
    client: APIClient,
    tool: dict,
    tool_category: str,
    tool_name: str,
    api_info: dict,
    example_prompt: str,
    example_prompt_follow: str,
    suggestion_prompt: str,
    suggestion_prompt_follow: str,
    rewrite_prompt: str,
    rewrite_prompt_follow: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
    episodes: int,
):
    # [논문 핵심 대응] DRAFT의 trial-and-error 핵심 루프.
    # 논문의 3단계를 반복 수행한다:
    # 1) Experience Gathering: Explorer가 다양한 질의/파라미터를 만들고 실제 API를 호출
    # 2) Learning from Experience: Analyzer가 상호작용 결과에서 문서 개선 포인트를 추출
    # 3) Documentation Rewriting: Rewriter가 설명을 갱신하고 다음 탐색 힌트를 생성
    # episode 예산을 다 쓰거나 adaptive termination이 발동하면 종료한다.
    api_name = api_info["name"]
    last_tool_description = api_info["description"]
    required_parameters = api_info["required_parameters"]
    optional_parameters = api_info["optional_parameters"]
    explored_queries = []
    explored_queries_embeddings = []
    explored_examples = []
    suggestions = []
    rewrite_description_history = [last_tool_description]
    rewrite_agent_history = []
    suggestion_from_rewrite_agent = ""

    for episode in range(episodes):
        # 현재 에피소드에서 사용할 API 설명 스냅샷(매 반복마다 갱신됨).
        tool_info = {
            "category": tool_category,
            "name": api_name,
            "description": rewrite_description_history[-1],
            "required_parameters": required_parameters,
            "optional_parameters": optional_parameters,
        }
        tool_description = str(tool_info)
        explore_prompt = example_prompt.replace("{Tool Description}", tool_description)

        if len(explored_queries) > 0:
            explore_prompt_follow_temp = example_prompt_follow.replace("{Explored queries}", str(explored_queries))
            explore_prompt_follow_temp = explore_prompt_follow_temp.replace(
                "{Suggestions}", suggestion_from_rewrite_agent
            )
            explore_prompt = explore_prompt + explore_prompt_follow_temp
            for _ in range(3):
                # Diversity-promoting exploration:
                # 임베딩 유사도(>0.9)가 높은 중복 질의를 거절하고 재생성시켜 탐색 다양성을 확보한다.
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": explore_prompt},
                ]
                example_ans = await openai_response(
                    client, messages, temperature, top_p, max_tokens, model, "Explorer"
                )
                cur_embedding = await openai_embedding(client, example_ans["User Query"])
                similarity = [
                    cosine_similarity(emb, cur_embedding)
                    for emb in explored_queries_embeddings
                ]
                if all(sim < 0.9 for sim in similarity):
                    break
                explore_prompt += (
                    f"\nYour last generate query '{example_ans['User Query']}' is too similar "
                    "to the previous ones. Please generate a different query."
                )
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": explore_prompt},
            ]
            example_ans = await openai_response(
                client, messages, temperature, top_p, max_tokens, model, "Explorer"
            )

        explored_queries.append(example_ans["User Query"])
        explored_queries_embeddings.append(await openai_embedding(client, example_ans["User Query"]))

        # 생성된 파라미터로 실제 도구를 호출해 self-driven interaction 신호를 획득한다.
        cate = tool_category
        tool_name_std = change_name(standardize(tool_name))
        api_name_std = change_name(standardize(api_name))
        parameters = example_ans["Parameters"]
        payload = {
            "category": cate,
            "tool_name": tool_name_std,
            "api_name": api_name_std,
            "tool_input": parameters,
            "strip": "filter",
            "rapidapi_key": rapidapi_key,
        }

        response = await asyncio.to_thread(get_rapidapi_response, payload)
        example_ans["API_Response"] = response
        explored_examples.append(example_ans)

        # Analyzer는 (설명, 질의, 파라미터, 실제 응답)에서 문서 개선 제안을 도출한다.
        suggestion_prompt_temp = suggestion_prompt.replace("{Tool Description}", tool_description)
        suggestion_prompt_temp = suggestion_prompt_temp.replace("{usage_example}", str(example_ans))
        if len(rewrite_description_history) > 1:
            suggestion_prompt_follow_temp = suggestion_prompt_follow.replace(
                "{History}", str(rewrite_description_history)
            )
            suggestion_prompt_temp += suggestion_prompt_follow_temp
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": suggestion_prompt_temp},
        ]
        suggestion_ans = await openai_response(
            client, messages, temperature, top_p, max_tokens, model, "Analyzer"
        )
        suggestions.append(suggestion_ans)

        # Rewriter는 Analyzer 피드백을 반영해
        # - API 설명(다음 에피소드의 입력)을 수정하고
        # - 다음 탐색을 위한 제안(Explorer 가이드)을 생성한다.
        rewrite_prompt_temp = rewrite_prompt.replace("{Tool Description}", tool_description)
        rewrite_prompt_temp = rewrite_prompt_temp.replace("{usage_example}", str(example_ans))
        rewrite_prompt_temp = rewrite_prompt_temp.replace(
            "{Suggestions}", suggestion_ans["Suggestions for tool description"]
        )
        rewrite_prompt_temp = rewrite_prompt_temp.replace("{tool_description}", tool_info["description"])
        if len(rewrite_description_history) > 1:
            rewrite_prompt_follow_temp = rewrite_prompt_follow.replace(
                "{History}", str(rewrite_description_history)
            )
            rewrite_prompt_temp += rewrite_prompt_follow_temp
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rewrite_prompt_temp},
        ]
        rewrtite_ans = await openai_response(
            client, messages, temperature, top_p, max_tokens, model, "Rewriter"
        )
        rewrite_description_history.append(rewrtite_ans["Rewritten description"])
        last_tool_description = rewrtite_ans["Rewritten description"]
        suggestion_from_rewrite_agent = str(rewrtite_ans["Suggestions for exploring"])
        rewrite_tool = {"tool_description": rewrtite_ans}
        rewrite_agent_history.append(rewrite_tool)

        if len(rewrite_description_history) > 1:
            # Tool-adaptive termination:
            # 연속된 두 설명이 지나치게 유사하면(의미+문자열) 수렴으로 판단해 조기 종료한다.
            # 불필요한 반복 수정을 줄여 효율을 높이고 과도한 편집을 방지한다.
            reference_sentence = rewrite_description_history[-2]
            candidate_sentence = rewrite_description_history[-1]
            delta = await compute_similarity_and_bleu(client, reference_sentence, candidate_sentence)
            if delta > 0.75:
                break

    # 최종 개선된 API 설명을 tool_guidelines에 반영.
    api_info["description"] = rewrite_description_history[-1]


async def process_tool(
    client: APIClient,
    tool: dict,
    example_prompt: str,
    example_prompt_follow: str,
    suggestion_prompt: str,
    suggestion_prompt_follow: str,
    rewrite_prompt: str,
    rewrite_prompt_follow: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
    episodes: int,
):
    # API 단위 개선: 하나의 tool 아래 각 API endpoint를 반복적으로 정제.
    tool_category = tool["category"]
    tool_name = tool["tool_name"]

    for _, api_info in tool["tool_guidelines"].items():
        await process_api_info(
            client,
            tool,
            tool_category,
            tool_name,
            api_info,
            example_prompt,
            example_prompt_follow,
            suggestion_prompt,
            suggestion_prompt_follow,
            rewrite_prompt,
            rewrite_prompt_follow,
            temperature,
            top_p,
            max_tokens,
            model,
            episodes,
        )

    # Tool 단위 재작성: API별 개선 후 전체 tool_description을 다시 생성.
    tool_doc = str(tool)

    with open("prompts/rewrite_tool_doc.txt", "r") as file:
        prompt_template = file.read()
    rewrite_tool_prompt = prompt_template.replace("{Tool Description}", tool_doc)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": rewrite_tool_prompt},
    ]

    tool_description_obj = await openai_response(
        client,
        messages,
        temperature,
        top_p,
        max_tokens,
        model,
        "ToolDoc",
    )
    tool["tool_description"] = tool_description_obj["tool_description"]
    return tool


async def main():
    # 프롬프트 3종은 논문의 3개 역할에 대응:
    # Explorer(경험 수집), Analyzer(경험 분석), Rewriter(문서 재작성).
    with open("prompts/Explorer.txt", "r") as file:
        example_prompt_template = file.read()
        example_prompt, example_prompt_follow = example_prompt_template.split("=========")

    with open("prompts/Analyzer.txt", "r") as file:
        suggestion_prompt_template = file.read()
        suggestion_prompt, suggestion_prompt_follow = suggestion_prompt_template.split("=========")

    with open("prompts/Rewriter.txt", "r") as file:
        rewrite_prompt_template = file.read()
        rewrite_prompt, rewrite_prompt_follow = rewrite_prompt_template.split("=========")

    with open("dataset/ToolBench/tool_instruction/Initial.json", "r", encoding="utf-8") as file:
        tools = json.load(file)

    temperature = 0.2
    top_p = 1
    max_tokens = 2000
    model = "gpt-4o-2024-08-06"
    episodes = 5

    # tool 단위 비동기 병렬 처리로 대규모 도구 집합에서도 self-driven 상호작용을 확장.
    client = APIClient(timeout_seconds=120, max_parallel=200)

    success_tools = []
    failed_tools = []

    tasks = []
    tool_keys = list(tools.keys())
    for key in tool_keys:
        tasks.append(
            process_tool(
                client,
                tools[key],
                example_prompt,
                example_prompt_follow,
                suggestion_prompt,
                suggestion_prompt_follow,
                rewrite_prompt,
                rewrite_prompt_follow,
                temperature,
                top_p,
                max_tokens,
                model,
                episodes,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for key, result in zip(tool_keys, results):
        if isinstance(result, Exception):
            failed_tools.append(
                {
                    "tool_id": key,
                    "tool": tools[key],
                    "error": str(result),
                    "retry_payload": {
                        "tool_id": key,
                        "category": tools[key].get("category"),
                        "tool_name": tools[key].get("tool_name"),
                    },
                }
            )
        else:
            success_tools.append(result)

    with open("DRAFT_success.json", "w", encoding="utf-8") as f:
        json.dump(success_tools, f, ensure_ascii=False, indent=4)

    with open("DRAFT_failed.json", "w", encoding="utf-8") as f:
        json.dump(failed_tools, f, ensure_ascii=False, indent=4)

    with open("DRAFT_retry_queue.json", "w", encoding="utf-8") as f:
        json.dump([item["retry_payload"] for item in failed_tools], f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
