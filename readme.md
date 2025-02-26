# Installation

To install and set up the  RAG system, follow these steps:

### **Prerequisites**

Ensure you have the following installed:

* Python 3.8 or higher
* pip (Python package manager)
* Git

### **Step 1: Clone the Repository**

```
git clone repo
cd rag
```

### **Step 2: Create a Virtual Environment**

```
python -m venv rag_demo
source rag_demo/bin/activate   # On macOS/Linux
rag_demo\Scripts\activate      # On Windows
```

### **Step 3: Install Dependencies**

```
pip install -r requirements.txt
```

### **Step 4: Set Up Environment Variables**

To configure the necessary API keys, create a `.env` file in the root directory and add the following variables:

#### 🔑 **Environment Variables**

| Variable           | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `TEVILY_API_KEY` | Used in **Corrective RAG** to fetch additional data from the web when the query and document are not relevant. |
| `GEMINI_API_KEY` | Required for generating responses using Google's Gemini LLM.                                                          |

#### 📌 **Example `.env` File**

```sh
TEVILY_API_KEY=your_tevily_api_key
GEMINI_API_KEY=your_gemini_api_key
```


## Step 5: Run the Code

You can specify the type of RAG (Retrieval-Augmented Generation) and optionally provide a query.

### Usage:

```sh
python main.py --rag=<rag_type> --query="your prompt"
```

### Parameters:

* `--rag`  *(required)* : Specifies the type of RAG to use. Available options:
  * `simple`
  * `multilevel`
  * `corrective`
  * `speculative`
  * `fusion`
* `--query`  *(optional)* : Your custom prompt.
  * If not provided, predefined test queries will be used.

# Running the RAG System

## 🔍 Examples

### ✅ Example 1: Run with a specific query

You can execute the script using the following command:

```sh
python main.py --rag=simple --query="What is mental health?"
```

### ✅ Example 2: Run with predefined test queries

```sh
python main.py --rag=simple
```

# Detail Research

# Types of Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) enhances AI models by integrating retrieval mechanisms with generative capabilities. This approach ensures responses are more accurate, contextually relevant, and informed by real-world data. Below are various RAG techniques, each designed to address different challenges and optimize AI-generated content.

## **1️⃣ Simple RAG (Baseline Retrieval-Augmented Generation)**

### **Description:**

Simple RAG is the most basic form of retrieval-augmented generation. When a user submits a query, the system retrieves relevant documents from a vector database (e.g., ChromaDB) based on semantic similarity. The retrieved documents, along with the original query, are then passed to an LLM (Large Language Model) to generate the final response.

### **Steps:**

1. Receive the  **user query** .
2. Retrieve the **most relevant document(s)** from ChromaDB.
3. Pass the **query + retrieved document(s)** to the LLM.
4. The  **LLM generates the final response** .

### **Pros:**

✔ Fast and simple to implement.
✔ Low computational cost.

### **Cons:**

✖ May return incorrect or incomplete information if the retrieval step is not accurate.
✖ Cannot verify or rank the retrieved documents.

## **2️⃣ Multilevel RAG (Ranked Context Retrieval)**

### **Description:**

Multilevel RAG introduces a **ranking mechanism** to improve retrieval quality. Instead of using all retrieved documents directly, the system first **asks the LLM to rank the context** based on relevance to the user query. Only the **top-ranked contexts** are used to generate the response, improving accuracy.

### **Steps:**

1. Receive the  **user query** .
2. Retrieve **multiple relevant documents** from ChromaDB.
3. Ask the **LLM to rank the retrieved documents** based on relevance.
4. Use the **top-ranked documents** as context.
5. Pass the **query + top-ranked documents** to the LLM.
6. The  **LLM generates the final response** .

### **Pros:**

✔ Improves response accuracy by filtering out low-quality context.
✔ Reduces hallucinations compared to Simple RAG.

### **Cons:**

✖ Requires an extra LLM call for ranking, increasing latency.
✖ Ranking by LLM may introduce biases.

## **3️⃣ Speculative RAG (Diverse Answer Generation)**

### **Description:**

Speculative RAG focuses on generating **multiple distinct responses** and selecting the best one. Instead of relying on a single LLM output, it encourages diversity by generating different answers and then ranking them to find the most relevant one.

### **Steps:**

1. Receive the  **user query** .
2. Retrieve **relevant documents** from ChromaDB.
3. Pass the **query + retrieved documents** to the LLM multiple times to generate  **distinct responses** .
4. Rank the responses based on relevance.
5. Select the **highest-ranked response** as the final answer.

### **Pros:**

✔ Reduces hallucinations by considering multiple perspectives.
✔ Useful for ambiguous or open-ended questions.

### **Cons:**

✖ Computationally expensive (multiple LLM calls).
✖ Requires an effective ranking system.

## **4️⃣ Fusion RAG (Query Expansion + Reciprocal Rank Fusion)**

### **Description:**

Fusion RAG enhances retrieval by **expanding the user query into multiple related questions** before fetching documents. It then applies a **Reciprocal Rank Fusion (RRF) algorithm** to rank the retrieved results, ensuring a more **diverse and comprehensive** context for the final response.

### **Steps:**

1. Receive the  **user query** .
2. Generate **multiple related sub-queries** based on the original query.
3. Retrieve **documents for each sub-query** from ChromaDB.
4. Apply **Reciprocal Rank Fusion (RRF) to rerank** and select the top 5 most relevant documents.
5. Pass the **query + top 5 documents** to the LLM.
6. The  **LLM generates the final response** .

### **Pros:**

✔ Expands retrieval scope, reducing missing information.
✔ Increases diversity in the retrieved context.

### **Cons:**

✖ Computationally expensive (multiple sub-queries and reranking).
✖ Can introduce irrelevant data if query expansion is inaccurate.

## **5️⃣ Corrective RAG (Verification & Web Search)**

### **Description:**

Corrective RAG  **verifies the retrieved documents before generating a response** . If the retrieved context is insufficient or inaccurate, it performs **external web searches** to supplement the missing information.

### **Steps:**

1. Receive the  **user query** .
2. Retrieve  **documents from ChromaDB** .
3. Perform  **grading (similarity check) between the query and retrieved documents** .
4. If similarity is  **high** , proceed with LLM response generation.
5. If similarity is  **low** , perform **external web search** to fetch additional information.
6. Pass the **query + validated/retrieved documents** to the LLM.
7. The  **LLM generates the final response** .

### **Pros:**

✔ Ensures high accuracy by validating the retrieved context.
✔ Reduces hallucinations by supplementing missing or incorrect information.

### **Cons:**

✖ Slowest method due to external web searches.
✖ Requires a dependency on web search APIs (Google Search, SerpAPI, etc.).

These metrics help in systematically evaluating and improving the quality of responses generated using RAG techniques

## 📊 Evaluation Metrics for RAG

| **Metric**                  | **Description**                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **BLEU Score**              | Measures the similarity between the generated response and a reference response by evaluating overlapping n-grams. A higher BLEU score indicates better linguistic similarity. |
| **ROUGE Scores**            | Compares the generated response with a reference response based on overlapping words and phrases.                                                                              |
| **ROUGE-1**                 | Measures the overlap of unigrams (single words) between the response and reference.                                                                                            |
| **ROUGE-2**                 | Measures the overlap of bigrams (two-word sequences) between the response and reference.                                                                                       |
| **ROUGE-L**                 | Evaluates the longest common subsequence between the response and reference.                                                                                                   |
| **Relevance**               | Measures how relevant the response is to the query. (0.00 = Not relevant, 1.00 = Fully relevant)                                                                               |
| **Accuracy**                | Evaluates the correctness of the response based on the given context. (0.00 = Incorrect, 1.00 = Perfectly accurate)                                                            |
| **Completeness**            | Assesses whether the response fully answers the query. (0.00 = Incomplete, 1.00 = Fully complete)                                                                              |
| **Context Precision**       | Determines if the response extracts precise information from the context. (0.00 = Not precise, 1.00 = Very precise)                                                            |
| **Context Recall**          | Checks if the response includes all key information from the context. (0.00 = Missing details, 1.00 = All details covered)                                                     |
| **Context Entities Recall** | Ensures important entities are mentioned correctly. (0.00 = Entities missing or incorrect, 1.00 = All entities correct)                                                        |
| **Noise Sensitivity**       | Measures how much irrelevant information is included in the response. (0.00 = No noise, 1.00 = Highly noisy)                                                                   |
| **Response Relevancy**      | Evaluates whether the response is well-structured and relevant. (0.00 = Irrelevant, 1.00 = Highly relevant)                                                                    |
| **Faithfulness**            | Determines whether the response avoids hallucination and stays true to the provided context. (0.00 = Hallucinated, 1.00 = Fully faithful)                                      |

# Statistical Report

## Simple RAG - Statistical Report

The following table provides a statistical evaluation of Simple RAG:

| Query                                                     | Truth                                             | LLM Response                                      | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ---- | ------- | ------- | ------- |
| What is mental health?                                    | Mental health encompasses our emotional, psych... | Mental health is our emotional, psychological,... | 1.00      | 0.98     | 0.95         | 0.95              | 0.92           | 0.98                    | 0.05              | 0.98               | 0.98         | 0.37 | 0.68    | 0.46    | 0.58    |
| What are some common mental health conditions?            | Common mental health conditions include a vari... | Common mental health conditions include: antis... | 1.00      | 1.00     | 1.00         | 1.00              | 1.00           | 1.00                    | 0.00              | 1.00               | 1.00         | 0.07 | 0.58    | 0.19    | 0.44    |
| What are the early warning signs of mental health issues? | Early warning signs of mental health problems ... | Early warning signs of mental health problems ... | 0.98      | 0.95     | 0.98         | 0.98              | 0.98           | 0.98                    | 0.05              | 0.98               | 0.98         | 0.25 | 0.55    | 0.38    | 0.48    |
| What is a Serious Mental Illness (SMI)?                   | A Serious Mental Illness (SMI) refers to a men... | A Serious Mental Illness (SMI) is a mental ill... | 0.98      | 0.95     | 0.92         | 0.95              | 0.90           | 0.95                    | 0.05              | 0.98               | 0.97         | 0.09 | 0.50    | 0.21    | 0.34    |
| What factors contribute to mental health conditions?      | Several factors contribute to the development ... | Mental health conditions are influenced by a c... | 1.00      | 1.00     | 0.80         | 1.00              | 0.80           | 1.00                    | 0.00              | 1.00               | 1.00         | 0.12 | 0.64    | 0.32    | 0.48    |

### **Average Simple RAG Analysis Report**

| Metric          | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| **Score** | 0.992     | 0.976    | 0.930        | 0.976             | 0.920          | 0.982                   | 0.030             | 0.988              | 0.986        | 0.183 | 0.592   | 0.317   | 0.468   |

## Multi-Level RAG - Statistical Report

The following table provides a statistical evaluation of Multi-Level RAG:

| Query                                                     | Truth                                             | LLM Response                                      | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ---- | ------- | ------- | ------- |
| What is mental health?                                    | Mental health encompasses our emotional, psych... | Mental health is our emotional, psychological,... | 1.00      | 0.98     | 0.96         | 0.98              | 0.95           | 0.98                    | 0.03              | 0.99               | 0.98         | 0.40 | 0.72    | 0.50    | 0.60    |
| What are some common mental health conditions?            | Common mental health conditions include a vari... | Common mental health conditions include: antis... | 1.00      | 1.00     | 1.00         | 1.00              | 1.00           | 1.00                    | 0.00              | 1.00               | 1.00         | 0.10 | 0.62    | 0.30    | 0.50    |
| What are the early warning signs of mental health issues? | Early warning signs of mental health problems ... | Early warning signs of mental health problems ... | 0.98      | 0.96     | 0.98         | 0.98              | 0.98           | 0.98                    | 0.02              | 0.99               | 0.99         | 0.28 | 0.58    | 0.42    | 0.50    |
| What is a Serious Mental Illness (SMI)?                   | A Serious Mental Illness (SMI) refers to a men... | A Serious Mental Illness (SMI) is a mental ill... | 0.98      | 0.95     | 0.94         | 0.96              | 0.91           | 0.95                    | 0.04              | 0.98               | 0.97         | 0.12 | 0.54    | 0.25    | 0.38    |
| What factors contribute to mental health conditions?      | Several factors contribute to the development ... | Mental health conditions are influenced by a c... | 1.00      | 1.00     | 0.90         | 1.00              | 0.85           | 1.00                    | 0.00              | 1.00               | 1.00         | 0.15 | 0.68    | 0.40    | 0.52    |

### Average Multi-Level RAG Analysis Report

| Metric          | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| **Score** | 0.982     | 0.966    | 0.934        | 0.964             | 0.920          | 0.964                   | 0.040             | 0.984              | 0.984        | 0.171 | 0.519   | 0.278   | 0.437   |

## Fusion RAG - Statistical Report

The following table provides a statistical evaluation of Fusion RAG:

| # | Query                                                     | Truth (Summary)                                   | LLM Response (Summary)                            | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L |
| - | --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ------ | ------- | ------- | ------- |
| 0 | What is mental health?                                    | Mental health encompasses our emotional, psych... | Mental health is a multifaceted concept encomp... | 0.95      | 0.90     | 0.92         | 0.90              | 0.90           | 0.92                    | 0.05              | 0.95               | 0.95         | 0.0505 | 0.3975  | 0.1635  | 0.2360  |
| 1 | What are some common mental health conditions?            | Common mental health conditions include a vari... | Common mental health conditions include eating... | 0.95      | 0.90     | 0.85         | 0.92              | 0.80           | 0.90                    | 0.15              | 0.90               | 0.95         | 0.1375 | 0.5111  | 0.2500  | 0.3556  |
| 2 | What are the early warning signs of mental health issues? | Early warning signs of mental health problems ... | Early warning signs of mental health problems ... | 0.95      | 0.90     | 0.85         | 0.92              | 0.80           | 0.85                    | 0.10              | 0.90               | 0.95         | 0.2639 | 0.5974  | 0.4533  | 0.5974  |
| 3 | What is a Serious Mental Illness (SMI)?                   | A Serious Mental Illness (SMI) refers to a men... | Serious Mental Illness (SMI) is a condition th... | 0.95      | 0.95     | 0.90         | 0.90              | 0.85           | 0.92                    | 0.05              | 0.98               | 0.98         | 0.1477 | 0.5474  | 0.3656  | 0.4421  |
| 4 | What factors contribute to mental health conditions?      | Several factors contribute to the development ... | Many factors contribute to mental health condi... | 0.95      | 0.95     | 0.85         | 0.90              | 0.90           | 0.95                    | 0.10              | 0.90               | 0.95         | 0.0946 | 0.4384  | 0.2817  | 0.4110  |

### **Average Fusion RAG Analysis Report**

| Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU     | ROUGE-1  | ROUGE-2  | ROUGE-L  |
| --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | -------- | -------- | -------- | -------- |
| 0.950     | 0.920    | 0.874        | 0.908             | 0.850          | 0.908                   | 0.090             | 0.926              | 0.956        | 0.138846 | 0.498351 | 0.302827 | 0.408409 |

## Speculative RAG - Statistical Report

The following table provides a statistical evaluation of Speculative RAG:

| Query                                                     | Truth                                             | LLM Response                                      | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| What is mental health?                                    | Mental health encompasses our emotional, psych... | Mental health includes our emotional, psycholo... | 1.0       | 0.95     | 0.6          | 0.9               | 0.6            | 0.6                     | 0.0               | 0.95               | 1.0          | 0.231 | 0.604   | 0.510   | 0.583   |
| What are some common mental health conditions?            | Common mental health conditions include a vari... | Some common mental health conditions include: ... | 1.0       | 1.00     | 1.0          | 1.0               | 1.0            | 1.0                     | 0.0               | 1.00               | 1.0          | 0.046 | 0.545   | 0.309   | 0.484   |
| What are the early warning signs of mental health issues? | Early warning signs of mental health problems ... | Experiencing one or more of the following feel... | 1.0       | 1.00     | 1.0          | 1.0               | 1.0            | 1.0                     | 0.0               | 1.00               | 1.0          | 0.005 | 0.295   | 0.088   | 0.240   |
| What is a Serious Mental Illness (SMI)?                   | A Serious Mental Illness (SMI) refers to a men... | A Serious Mental Illness (SMI) is a mental ill... | 1.0       | 1.00     | 0.8          | 1.0               | 0.6            | 0.0                     | 0.0               | 1.00               | 1.0          | 0.126 | 0.574   | 0.339   | 0.462   |
| What factors contribute to mental health conditions?      | Several factors contribute to the development ... | Biological factors, such as genes or brain che... | 1.0       | 1.00     | 0.8          | 1.0               | 0.8            | 1.0                     | 0.0               | 1.00               | 1.0          | 0.098 | 0.521   | 0.447   | 0.521   |

### Average Speculative RAG Analysis Report

| Metric | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------ | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| Score  | 1.000     | 0.990    | 0.840        | 0.980             | 0.800          | 0.720                   | 0.000             | 0.990              | 1.000        | 0.102 | 0.508   | 0.339   | 0.459   |

## Corrective RAG - Statistical Report

The following table provides a statistical evaluation of Corrective RAG:

| Query                                                     | Truth                                             | LLM Response                                      | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| What is mental health?                                    | Mental health encompasses our emotional, psych... | Mental health encompasses emotional, psycholog... | 1.00      | 0.95     | 0.95         | 0.98              | 0.95           | 0.98                    | 0.05              | 0.98               | 0.98         | 0.149 | 0.487   | 0.277   | 0.396   |
| What are some common mental health conditions?            | Common mental health conditions include a vari... | Common mental health conditions encompass a wi... | 1.00      | 0.98     | 0.95         | 0.99              | 0.95           | 0.98                    | 0.05              | 0.98               | 0.99         | 0.039 | 0.247   | 0.137   | 0.185   |
| What are the early warning signs of mental health issues? | Early warning signs of mental health problems ... | Early warning signs of mental health problems ... | 1.00      | 0.98     | 0.99         | 0.99              | 0.98           | 1.00                    | 0.01              | 0.99               | 1.00         | 0.034 | 0.195   | 0.079   | 0.167   |
| What is a Serious Mental Illness (SMI)?                   | A Serious Mental Illness (SMI) refers to a men... | A Serious Mental Illness (SMI) is a mental hea... | 0.98      | 0.95     | 0.90         | 0.92              | 0.85           | 0.90                    | 0.15              | 0.97               | 0.98         | 0.064 | 0.313   | 0.151   | 0.201   |
| What factors contribute to mental health conditions?      | Several factors contribute to the development ... | Factors contributing to mental health conditio... | 0.98      | 0.95     | 0.97         | 0.96              | 0.94           | 0.95                    | 0.05              | 0.99               | 0.98         | 0.025 | 0.168   | 0.090   | 0.123   |

### Average Corrective RAG Analysis Report

| Metric | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Context Entities Recall | Noise Sensitivity | Response Relevancy | Faithfulness | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------ | --------- | -------- | ------------ | ----------------- | -------------- | ----------------------- | ----------------- | ------------------ | ------------ | ----- | ------- | ------- | ------- |
| Score  | 0.992     | 0.962    | 0.952        | 0.968             | 0.934          | 0.962                   | 0.062             | 0.982              | 0.986        | 0.062 | 0.282   | 0.146   | 0.214   |

## Top 5 RAG Models Based on Categories Overall

| RAG Model       | Relevance | Accuracy | Completeness | Context Precision | Context Recall | Faithfulness |
| --------------- | --------- | -------- | ------------ | ----------------- | -------------- | ------------ |
| Speculative RAG | 1.000     | 0.990    | 0.840        | 0.980             | 0.800          | 1.000        |
| Corrective RAG  | 0.992     | 0.962    | 0.952        | 0.968             | 0.934          | 0.986        |
| Simple RAG      | 0.992     | 0.976    | 0.930        | 0.976             | 0.920          | 0.986        |
| Multi-Level RAG | 0.982     | 0.966    | 0.934        | 0.964             | 0.920          | 0.984        |
| Fusion RAG      | 0.950     | 0.920    | 0.874        | 0.908             | 0.850          | 0.956        |

## Top 5 RAG Models for Each Evaluation Metric

### **Relevance**

| Rank | RAG Model       | Score |
| ---- | --------------- | ----- |
| 1    | Speculative RAG | 1.000 |
| 2    | Corrective RAG  | 0.992 |
| 3    | Simple RAG      | 0.992 |
| 4    | Multi-Level RAG | 0.982 |
| 5    | Fusion RAG      | 0.950 |

### **Accuracy**

| Rank | RAG Model       | Score |
| ---- | --------------- | ----- |
| 1    | Speculative RAG | 0.990 |
| 2    | Simple RAG      | 0.976 |
| 3    | Multi-Level RAG | 0.966 |
| 4    | Corrective RAG  | 0.962 |
| 5    | Fusion RAG      | 0.920 |

### **Completeness**

| Rank | RAG Model       | Score |
| ---- | --------------- | ----- |
| 1    | Corrective RAG  | 0.952 |
| 2    | Multi-Level RAG | 0.934 |
| 3    | Simple RAG      | 0.930 |
| 4    | Fusion RAG      | 0.874 |
| 5    | Speculative RAG | 0.840 |

### **Faithfulness**

| Rank | RAG Model       | Score |
| ---- | --------------- | ----- |
| 1    | Speculative RAG | 1.000 |
| 2    | Corrective RAG  | 0.986 |
| 3    | Simple RAG      | 0.986 |
| 4    | Multi-Level RAG | 0.984 |
| 5    | Fusion RAG      | 0.956 |

### **BLEU Score**

| Rank | RAG Model       | Score |
| ---- | --------------- | ----- |
| 1    | Simple RAG      | 0.183 |
| 2    | Multi-Level RAG | 0.171 |
| 3    | Fusion RAG      | 0.139 |
| 4    | Speculative RAG | 0.102 |
| 5    | Corrective RAG  | 0.062 |

## Final Verdict

🔹 If you need **speed** → **Simple RAG**

🔹 If you need **better-ranked context** → **Multilevel RAG**

🔹 If you need **diverse answers** → **Speculative RAG**

🔹 If you need **broad context & better ranking** → **Fusion RAG**

🔹 If you need **highest accuracy & verification** → **Corrective RAG** ✅

### **Final Verdict:** **Speculative RAG** leads in overall performance, but **Corrective RAG** and **Simple RAG** offer the best balance of accuracy, completeness, and faithfulness. 🚀✅
