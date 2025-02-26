 
from collections import defaultdict
from litellm import completion 
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import json
from typing import Dict
from itertools import chain
from tavily import TavilyClient
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


load_dotenv()


MODEL = "gemini/gemini-1.5-flash-8b"
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


client = chromadb.PersistentClient(path="db")
collection_name = "mental"
collection = client.get_or_create_collection(name=collection_name)


def add_data_to_chroma(path):
    """this method is use to store the data from txt file"""
    with open(path,"rb") as f:
        filedata = f.read()
        file_string = filedata.decode() 
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )      
    chunks = splitter.split_text(file_string)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        ids = ids,
        documents=chunks)
    print("Data Added Successfully")
    

def fetch_documents(prompt: str):
    """Retrieve relevant documents from ChromaDB"""
    results = collection.query(query_texts=[prompt], n_results=2)
    res = results["documents"][0]
    context = "\n".join(res)
    return context


def simple_rag(query:str):
    context  = fetch_documents(query)
    messages = [
        {'role': 'system', 'content': f'You  are an AI assistat made to give reponses to the user queries based on the provided context. Please do not answer any question out of context. Please provide information to the user in 5 to 10 lines. CONTEXT: {context}'},
        {'role': 'user', 'content': query}
    ]

    response  = completion(
        messages=messages,
        api_key=GOOGLE_API_KEY,
        model=MODEL
    )
    
    score = evaluate_rag(query=query,response=str(response.choices[0].message.content),context=context)
    return score,response.choices[0].message.content
    

def multilevel_rag(query:str):
    context  = fetch_documents(query)
    messages = [
                    {
                "role": "user",
                "content": 
                    f"""You are an AI assistant that specializes in ranking and evalute relevant context based on user queries. Your task is to carefully evaluate all the provided contexts, rerank them based on their semantic similarity and relevance to the user query, and return the results as a JSON list. Each entry in the list should include the context and its corresponding rank, ensuring that the most relevant contexts are prioritized. If no relevant contexts are found, return an empty list. 
                    
                    CONTEXT : {context}
                    
                    OUTPUT
                    json(
                        'contexts': list(str) <List of the reordered contexts>
                        )
                        
                    """
            }
        ]   
    response = completion(
                model=MODEL,
                messages=messages,
                api_key=GOOGLE_API_KEY,
                response_type = {'type': 'json_object'}
            )
    ranked_context = response.choices[0].message.content
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that takes the ranked context provided by a previous model and generates a "
                "coherent and relevant response based on that context. Your task is to ensure that the response is "
                "clear, concise, and directly addresses the user's query while utilizing the provided context effectively."
                "Please only answer the query if its related to the context else respond with an apology, that you dont know the answer"
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on the ranked context provided: {ranked_context}, which includes its ranking, can you summarize "
                f"the key points and provide a clear answer to my question: {query}?"
            )
        }
    ]
    response = completion(
        model=MODEL,
        messages=messages,
        api_key=GOOGLE_API_KEY
    )
    
    score = evaluate_rag(query=query,response=str(response.choices[0].message.content),context=context)
    return score,response.choices[0].message.content


def speculative_rag(query: str, num_responses: int = 3) -> str:
    context  = fetch_documents(query)

    responses = []

    for _ in range(num_responses):
        # Generate a response while considering previous responses
        messages = [
            {"role": "system", "content": (
                "You are an AI assistant designed to generate relevant questions based on the provided context and user query.\n"
                "Follow these rules strictly:\n"
                "- Only generate a answer if the user query is relevant to the given context.\n"
                "- If the query is NOT related to the context, respond with: 'I cannot answer this question.'\n"
                f"- Context: {context}\n"
                f"- Previously generated responses: {responses if responses else 'None yet'}\n"
                "NOTE: Do NOT answer any question out of context. Stay strictly within the given context."
            )},
            {"role": "user", "content": query}
        ]

        response = completion(model=MODEL, api_key=GOOGLE_API_KEY, messages=messages)
        responses.append(response["choices"][0]["message"]["content"])

    eval_messages = [
    {"role": "system", "content": (
        "You are an AI assistant designed to evaluate responses generated by an LLM based on their relevance to a given user query.\n"
        "Your task is to select the most relevant response from the provided list.\n"
        "Follow these rules strictly:\n"
        "- Only consider responses that are directly relevant to the user query.\n"
        "- Ignore any responses that go off-topic or do not match the context.\n"
        "- If all responses are off-topic, return: 'I cannot answer this question.'\n"
    )},
    {"role": "user", "content": (
        "Below is a list of responses generated by the LLM:\n"
        f"{responses}\n\n"
        "User Query:\n"
        f"{query}\n\n"
        "Select the best response that most accurately answers the query.\n"
        "If no response is relevant, return: 'I cannot answer this question.'\n"
        "Note: RETURN ONLY THE SELECTED RESPONSE TEXT. DO NOT SCORE OR INCLUDE ANY OTHER INFORMATION."
    )}
]
    final_response = completion(model=MODEL, api_key=GOOGLE_API_KEY, messages=eval_messages)
    score = evaluate_rag(query=query,response=str(response.choices[0].message.content),context=context)
    return score,final_response["choices"][0]["message"]["content"]


def fusion_rag(query: str):
    """ Perform Fusion RAG with query expansion, retrieval, re-ranking, and final response generation with conflict resolution """

    # Step 1: Generate multiple query variations
    prompt_variation = f"""
    You are an expert search query generator. Given the following user question, generate **four** diverse and effective search queries that capture different aspects of the original query.

    - Each query should be **concise yet specific** to retrieve high-quality results.
    - Ensure **diversity** by covering multiple perspectives or subtopics.
    - Do **not** repeat queries with minor variations.
    - Output **exactly** 4 queries, each on a **new line**, without numbering or bullet points.

    User Question: "{query}"
    """

    response_variation = completion(
        model=MODEL,
        api_key=GOOGLE_API_KEY,
        messages=[{"role": "user", "content": prompt_variation}]
    )

    query_variations = response_variation["choices"][0]["message"]["content"].strip().split("\n")[:4]

    # Step 2: Retrieve relevant documents
    all_results = []
    for q in query_variations:
        results = collection.query(
            query_texts=[q],
            n_results=3  # Fetch top 3 results per query
        )
        all_results.append(results["documents"])
    
    retrieved_docs = list(chain(*all_results))  # Flatten list

    # Step 3: Apply Reciprocal Rank Fusion (RRF)
    fused_scores = defaultdict(float)
    k = 60

    for docs in retrieved_docs:
        for rank, doc in enumerate(docs):
            doc_str = doc  
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_docs = [doc[0] for doc in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

    # Step 4: Detect conflicting information
    context = "\n".join(reranked_docs[:6])  # Use top 6 most relevant documents

    conflict_resolution_prompt = f"""
    You are an **expert fact-checking AI** that analyzes and compares multiple sources of information to identify inconsistencies or contradictions. 

    **Task:**  
    - Carefully analyze the **six** retrieved documents provided below.  
    - Identify any **conflicting statements, differing perspectives, or factual inconsistencies** between them.  
    - If there are conflicts, **clearly summarize each viewpoint** and provide a **balanced, evidence-based conclusion**.  
    - If there are **no contradictions**, summarize the key insights from all sources into a well-structured response.  
    - Ensure your answer is **neutral, unbiased, and factually accurate**.

    **Retrieved Documents:**
    {context}

    **Instructions for Output:**  
    - If conflicts exist, structure your answer as follows:  
      1. **Conflicting Perspective 1:** [Summary]  
      2. **Conflicting Perspective 2:** [Summary]  
      3. **Balanced Conclusion:** [Final Answer]  
    - If there are **no conflicts**, simply provide a **concise summary** covering all the key information.
    """

    conflict_response = completion(
        model=MODEL,
        api_key=GOOGLE_API_KEY,
        messages=[{"role": "user", "content": conflict_resolution_prompt}]
    )

    resolved_context = conflict_response["choices"][0]["message"]["content"]

    # Step 5: Generate final response with conflict-aware context
    prompt_final = f"""
    You are an AI assistant tasked with answering user questions using the best available information while ensuring **accuracy, clarity, and neutrality**.

    **Instructions:**  
    - Read the context below carefully.  
    - generates a coherent and relevant response based on that context. Your task is to ensure that the response is clear, concise, and directly addresses the user's query while utilizing the provided context effectively."
    - If the information includes **multiple viewpoints**, fairly represent all perspectives before providing a final, balanced answer.  
    - Your response should be **detailed yet concise**, using **clear and structured language**.  
    - Avoid making assumptions or adding speculative details.
    - Please only answer the query if its related to the context else respond with an apology, that you dont know the answer
    
    **Context Information:**  
    {resolved_context}

    **User Question:**  
    {query}

    **Final Answer:**
    """

    response_final = completion(
        model=MODEL,
        api_key=GOOGLE_API_KEY,
        messages=[{"role": "user", "content": prompt_final}]
    )
    score = evaluate_rag(query=query,response=str(response_final.choices[0].message.content),context=context)
    return score,response_final["choices"][0]["message"]["content"]
        

def corrective_rag(query:str):
    """Perform Corrective RAG: Search, Generate, Grade, Correct, and Refine in a single function."""
    context  = fetch_documents(query)
    tavily_client = TavilyClient(api_key=os.getenv("TEVILY_API_KEY"))
    
    
    
    grading_prompt = f"""
        You are an expert evaluator responsible for assessing the relevance and sufficiency of the retrieved context in relation to the user query.
        Your task is to analyze whether the retrieved context provides enough accurate and relevant information to answer the query.

        ### **User Query:**
        {query}

        ### **Retrieved Context:**
        {context}

        ### **Evaluation Criteria:**
        - **[CORRECT]** (Score: 3) → The retrieved context fully aligns with the user query, providing sufficient and accurate details.
        - **[PARTIALLY CORRECT]** (Score: 2) → The retrieved context is somewhat relevant but lacks key details to fully answer the query.
        - **[INCORRECT]** (Score: 1) → The retrieved context is irrelevant, contains misinformation, or does not support answering the query.

        ### **Rules for Evaluation:**
        1. **Analyze** how well the retrieved context supports answering the user query.
        2. **Select one confidence level** from [CORRECT], [PARTIALLY CORRECT], or [INCORRECT].
        3. **Provide a score (1-3)** based on the relevance and completeness of the retrieved context.
        4. **Justify your decision** by highlighting what is correct, missing, or misleading in the context.

        ### **Expected Output Format:**
        Return the confidence level and a justification in the following format:

        **Example Output:**
        - **CORRECT (3)**: The retrieved context fully covers all aspects of the user query.
        - **PARTIALLY CORRECT (2)**: The context provides some relevant information but is missing details on [specific aspect].
        - **INCORRECT (1)**: The context does not align with the user query and lacks relevant information.

        Now, provide your evaluation.
    """


    grading_feedback = completion(
        model=MODEL,
        api_key=GOOGLE_API_KEY,
        messages=[{"role": "user", "content": grading_prompt}]
    )["choices"][0]["message"]["content"]
    
  
    grading_feedback_lower = grading_feedback.lower()
    
    # Step 4: If response is inaccurate, refine using web search
    if "incorrect (1)" in grading_feedback_lower:
        refined_context = None
        score = {
            "relevance":0,
            "accuracy":0,
            "completeness":0,
            "context_precision":0,
            "context_recall":0,
            "context_entities_recall":0,
            "noise_sensitivity":0,
            "response_relevancy":0,
            "faithfulness":0
        }
        return score, "Sorry I do not have access to that information."

    elif "partially correct (2)" in grading_feedback_lower:
        print("Fetching Data from Web Sources")
        data = tavily_client.search(query=query + " latest updates", num_results=3)
        contents = [result["content"] for result in data["results"]]
        refined_context = "\n".join(contents) + "\n\nThis is your old context:\n" + context
    
    else:
        refined_context = context
        
        
    refined_prompt = f"""
                You are an expert AI assistant responsible for generating the most **accurate, complete, and well-structured** response to a user query.

                ### **Task Overview**
                The initial response was evaluated and found to be: **{grading_feedback}**.  
                Your task is to **improve and refine** the response using the additional retrieved information.

                ### **User Query**  
                🔹 **Question Asked by the User:**  
                {query}  

                ### **Reference Material**
                🔹 **Updated Context (Additional Information Retrieved)**  
                {refined_context}  

                ### **Refinement Instructions**
                - **Ensure the response directly addresses the user query.**  
                - **If the initial response lacked details**, integrate the missing but relevant information.  
                - **If inaccuracies were found**, correct them while maintaining coherence.  
                - **Ensure factual correctness and clarity.**  
                - **Do not introduce hallucinated or speculative information.**  
                - **Maintain a professional and authoritative tone.**  

                ### **Expected Output**
                - **Final Corrected Response:** A well-structured, factually accurate, and refined answer based on the updated context.

                Now, generate the **corrected and improved response** based on the retrieved information and the user’s query.
            """ 


        # Generating response from model
    final_response = completion(
            model=MODEL,
            api_key=GOOGLE_API_KEY,
            messages=[{"role": "user", "content": refined_prompt}]
        )

    model_response = final_response.choices[0].message.content.strip() if final_response.choices else "Sorry I do not have access to that information."

        # If refined_context was None, enforce the correct response
    if refined_context is None and model_response.lower() != "sorry i do not have access to that information.":
        model_response = "Sorry I do not have access to that information."

        # Ensure function always returns a tuple
    score = evaluate_rag(query=query, response=model_response, context=context)

    return score, model_response




def evaluate_rag(query: str, response: str, context: str) -> Dict[str, float]:
    evaluation_prompt = f"""
    Evaluate the response based on the given query and context.
    
    Query: {query}
    Context: {context}
    Response: {response}
    
    Assign a score between **0.00 and 1.00 (two decimal places only)** for each metric.
    
    Metrics:
    - **Relevance**: How relevant is the response to the query? (0.00 = Not relevant, 1.00 = Fully relevant)
    - **Accuracy**: How accurate is the response based on the given context? (0.00 = Incorrect, 1.00 = Perfectly accurate)
    - **Completeness**: Does the response fully answer the query? (0.00 = Incomplete, 1.00 = Fully complete)
    - **Context Precision**: Does the response extract precise information from the context? (0.00 = Not precise, 1.00 = Very precise)
    - **Context Recall**: Does the response include all key information from the context? (0.00 = Missing details, 1.00 = All details covered)
    - **Context Entities Recall**: Are important entities mentioned correctly? (0.00 = Entities missing or incorrect, 1.00 = All entities correct)
    - **Noise Sensitivity**: How much irrelevant information is in the response? (0.00 = No noise, 1.00 = Highly noisy)
    - **Response Relevancy**: Is the response well-structured and relevant? (0.00 = Irrelevant, 1.00 = Highly relevant)
    - **Faithfulness**: Does the response avoid hallucination and stay true to the provided context? (0.00 = Hallucinated, 1.00 = Fully faithful)
    
    **Important: The output must be a JSON object with decimal scores (e.g., "accuracy": 0.87).**
    
    Example Output:
    {{
        "relevance": 0.92, 
        "accuracy": 0.85, 
        "completeness": 0.95, 
        "context_precision": 0.91, 
        "context_recall": 0.88, 
        "context_entities_recall": 0.90, 
        "noise_sensitivity": 0.22, 
        "response_relevancy": 0.93, 
        "faithfulness": 0.94
    }}
    """

    evaluation_result = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant that evaluates text responses based on given criteria."},
            {"role": "user", "content": evaluation_prompt}
        ],
        api_key=GOOGLE_API_KEY,
        response_format={"type": "json_object"}
    )

    try:
        scores = json.loads(evaluation_result["choices"][0]["message"]["content"])
        # Format all scores to 2 decimal places
        scores = {key: round(float(value), 2) for key, value in scores.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        scores = {
            "relevance": 0.00, "accuracy": 0.00, "completeness": 0.00,
            "context_precision": 0.00, "context_recall": 0.00, "context_entities_recall": 0.00,
            "noise_sensitivity": 1.00,  # Higher means more noise, should be lower ideally
            "response_relevancy": 0.00, "faithfulness": 0.00
        }
    return scores


def calculate_bleu_rouge(response: str, groundtruth: str):
    # Tokenize the response and groundtruth
    response_tokens = response.split()
    groundtruth_tokens = groundtruth.split()
    
    # Compute BLEU score
    smoothie = SmoothingFunction().method1  # To avoid zero scores for short sentences
    bleu_score = sentence_bleu([groundtruth_tokens], response_tokens, smoothing_function=smoothie)
    
    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(groundtruth, response)
    
    return {
        "bleu": bleu_score,
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure
    }

