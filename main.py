from rag import *
import pandas as pd
import time 
import sys
import argparse


parser = argparse.ArgumentParser(description="Process command-line query.")

# Add the --query argument
parser.add_argument("--query", type=str,required=False)
parser.add_argument("--rag", type=str,required=True)

# Parse arguments
args = parser.parse_args()

# Access the query argument
user_input = args.query
rag_name = args.rag


if user_input:
    choice = rag_name
    match choice:
        case "simple":
            user_query = user_input
            score,response = simple_rag(user_query)
            print("\n***********************************")
            print("\nLLM response :- ",response)
            print("*********\tScore\t*********\n")
            print("Relevance - ",score['relevance'])
            print("Accuracy - ",score['accuracy'])
            print("Completeness - ",score['completeness'])
            print("Context Precision - ",score['context_precision'])
            print("Context Recall - ",score['context_recall'])
            print("Context Entities Recall - ",score['context_entities_recall'])
            print("Noise Sensitivity - ",score['noise_sensitivity'])
            print("Response Relevancy - ",score['response_relevancy'])
            print("Faithfulness - ",score['faithfulness'])
        
        case "multilevel":
            user_query = user_input
            score,response = multilevel_rag(user_query)
            print("\n***********************************")
            print("\nLLM response :- ",response)
            print("*********\tScore\t*********")
            print("Relevance - ",score['relevance'])
            print("Accuracy - ",score['accuracy'])
            print("Completeness - ",score['completeness'])
            print("Context Precision - ",score['context_precision'])
            print("Context Recall - ",score['context_recall'])
            print("Context Entities Recall - ",score['context_entities_recall'])
            print("Noise Sensitivity - ",score['noise_sensitivity'])
            print("Response Relevancy - ",score['response_relevancy'])
            print("Faithfulness - ",score['faithfulness'])
            
            
        case "speculative":
            user_query = user_input
            score,response = speculative_rag(user_query)
            print("\n***********************************")
            print("\nLLM response :- ",response)
            print("*********\tScore\t*********")
            print("Relevance - ",score['relevance'])
            print("Accuracy - ",score['accuracy'])
            print("Completeness - ",score['completeness'])
            print("Context Precision - ",score['context_precision'])
            print("Context Recall - ",score['context_recall'])
            print("Context Entities Recall - ",score['context_entities_recall'])
            print("Noise Sensitivity - ",score['noise_sensitivity'])
            print("Response Relevancy - ",score['response_relevancy'])
            print("Faithfulness - ",score['faithfulness'])
            
        case "fusion":
            user_query = user_input
            score,response = fusion_rag(user_query)
            print("\n***********************************")
            print("\nLLM response :- ",response)
            print("*********\tScore\t*********")
            print("Relevance - ",score['relevance'])
            print("Accuracy - ",score['accuracy'])
            print("Completeness - ",score['completeness'])
            print("Context Precision - ",score['context_precision'])
            print("Context Recall - ",score['context_recall'])
            print("Context Entities Recall - ",score['context_entities_recall'])
            print("Noise Sensitivity - ",score['noise_sensitivity'])
            print("Response Relevancy - ",score['response_relevancy'])
            print("Faithfulness - ",score['faithfulness'])
            
            
        case "corrective":
            user_query = user_input
            score,response = corrective_rag(user_query)
            print("\n***********************************")
            print("\nLLM response :- ",response)
            print("*********\tScore\t*********")
            print("Relevance - ",score['relevance'])
            print("Accuracy - ",score['accuracy'])
            print("Completeness - ",score['completeness'])
            print("Context Precision - ",score['context_precision'])
            print("Context Recall - ",score['context_recall'])
            print("Context Entities Recall - ",score['context_entities_recall'])
            print("Noise Sensitivity - ",score['noise_sensitivity'])
            print("Response Relevancy - ",score['response_relevancy'])
            print("Faithfulness - ",score['faithfulness'])
        
        case _:
            print("Invalid input")

else:
    choice = rag_name    
    match choice:
        case "simple": 
            df = pd.read_csv("test.csv")
            data = []
            for i in range(len(df)):
                score,response = simple_rag(query=df["query"][i])
                embed_score = calculate_bleu_rouge(response=response,groundtruth=df['truth'][i])
                print("\n******************************")
                print('query :- ', df['query'][i])
                print('truth :- ', df['truth'][i])
                print('\nLLM_response :- ', response) 
                print("*******\t Score \t***********") 
                print('relevance :- ', score['relevance']) 
                print('accuracy :- ', score['accuracy'])
                print('completeness :- ', score['completeness'])
                print('context_precision :- ', score['context_precision'])
                print('context_recall :- ', score['context_recall'])
                print('context_entities_recall :- ', score['context_entities_recall'])
                print('noise_sensitivity :- ', score['noise_sensitivity'])
                print('response_relevancy :- ', score['response_relevancy'])
                print('faithfulness :- ', score['faithfulness'])
                print('bleu :- ', embed_score['bleu'])
                print('rouge1 :- ', embed_score['rouge1'])
                print('rouge2 :- ', embed_score['rouge2'])
                print('rougeL :- ', embed_score['rougeL'])

                row = {
                'query': df['query'][i],
                'truth': df['truth'][i],
                '\nLLM_response': response,  
                'relevance': score['relevance'], 
                'accuracy': score['accuracy'],
                'completeness': score['completeness'],
                'context_precision': score['context_precision'],
                'context_recall': score['context_recall'],
                'context_entities_recall': score['context_entities_recall'],
                'noise_sensitivity': score['noise_sensitivity'],
                'response_relevancy': score['response_relevancy'],
                'faithfulness': score['faithfulness'],
                'bleu': embed_score['bleu'],  
                'rouge1': embed_score['rouge1'],
                'rouge2': embed_score['rouge2'],
                'rougeL': embed_score['rougeL']
                }
            
                data.append(row)
            simple_rag_df = pd.DataFrame(data)
                                
            print("\nthis is a Simple Rag Average Data Report")
            numeric_columns = simple_rag_df.select_dtypes(include=['float64', 'int64'])
            average_values_simple = numeric_columns.mean()
            print(average_values_simple)
            
        case "multilevel": 
            df = pd.read_csv("test.csv")
            data = []
            for i in range(len(df)):
                print("\nProcessing Question",i+1)
                time.sleep(15)
                score,response = multilevel_rag(query=df["query"][i])
                embed_score = calculate_bleu_rouge(response=response,groundtruth=df['truth'][i])
                print("\n******************************")
                print('query :- ', df['query'][i])
                print('truth :- ', df['truth'][i])
                print('\nLLM_response :- ', response) 
                print("*******\t Score \t***********") 
                print('relevance :- ', score['relevance']) 
                print('accuracy :- ', score['accuracy'])
                print('completeness :- ', score['completeness'])
                print('context_precision :- ', score['context_precision'])
                print('context_recall :- ', score['context_recall'])
                print('context_entities_recall :- ', score['context_entities_recall'])
                print('noise_sensitivity :- ', score['noise_sensitivity'])
                print('response_relevancy :- ', score['response_relevancy'])
                print('faithfulness :- ', score['faithfulness'])
                print('bleu :- ', embed_score['bleu'])
                print('rouge1 :- ', embed_score['rouge1'])
                print('rouge2 :- ', embed_score['rouge2'])
                print('rougeL :- ', embed_score['rougeL'])

                row = {
                'query': df['query'][i],
                'truth': df['truth'][i],
                '\nLLM_response': response,  
                'relevance': score['relevance'], 
                'accuracy': score['accuracy'],
                'completeness': score['completeness'],
                'context_precision': score['context_precision'],
                'context_recall': score['context_recall'],
                'context_entities_recall': score['context_entities_recall'],
                'noise_sensitivity': score['noise_sensitivity'],
                'response_relevancy': score['response_relevancy'],
                'faithfulness': score['faithfulness'],
                'bleu': embed_score['bleu'],  
                'rouge1': embed_score['rouge1'],
                'rouge2': embed_score['rouge2'],
                'rougeL': embed_score['rougeL']
                }
            
                data.append(row)
            multilevel_rag_df = pd.DataFrame(data)
                                
            print("\nthis is a Multi Level Rag Average Data Report")
            numeric_columns = multilevel_rag_df.select_dtypes(include=['float64', 'int64'])
            average_values_simple = numeric_columns.mean()
            print(average_values_simple)
            
        case "speculative": 
            df = pd.read_csv("test.csv")
            data = []
            for i in range(len(df)):
                print("Processing Question",i+1)
                time.sleep(30)
                score,response = speculative_rag(query=df["query"][i])
                embed_score = calculate_bleu_rouge(response=response,groundtruth=df['truth'][i])
                print("\n******************************")
                print('query :- ', df['query'][i])
                print('truth :- ', df['truth'][i])
                print('\nLLM_response :- ', response) 
                print("*******\t Score \t***********") 
                print('relevance :- ', score['relevance']) 
                print('accuracy :- ', score['accuracy'])
                print('completeness :- ', score['completeness'])
                print('context_precision :- ', score['context_precision'])
                print('context_recall :- ', score['context_recall'])
                print('context_entities_recall :- ', score['context_entities_recall'])
                print('noise_sensitivity :- ', score['noise_sensitivity'])
                print('response_relevancy :- ', score['response_relevancy'])
                print('faithfulness :- ', score['faithfulness'])
                print('bleu :- ', embed_score['bleu'])
                print('rouge1 :- ', embed_score['rouge1'])
                print('rouge2 :- ', embed_score['rouge2'])
                print('rougeL :- ', embed_score['rougeL'])

                row = {
                'query': df['query'][i],
                'truth': df['truth'][i],
                '\nLLM_response': response,  
                'relevance': score['relevance'], 
                'accuracy': score['accuracy'],
                'completeness': score['completeness'],
                'context_precision': score['context_precision'],
                'context_recall': score['context_recall'],
                'context_entities_recall': score['context_entities_recall'],
                'noise_sensitivity': score['noise_sensitivity'],
                'response_relevancy': score['response_relevancy'],
                'faithfulness': score['faithfulness'],
                'bleu': embed_score['bleu'],  
                'rouge1': embed_score['rouge1'],
                'rouge2': embed_score['rouge2'],
                'rougeL': embed_score['rougeL']
                }
            
                data.append(row)
            speculative_rag_df = pd.DataFrame(data)
                                
            print("\nthis is a Speculative Level Rag Average Data Report")
            numeric_columns = speculative_rag_df.select_dtypes(include=['float64', 'int64'])
            average_values_simple = numeric_columns.mean()
            print(average_values_simple)
            
            
        case "fusion": 
            df = pd.read_csv("test.csv")
            data = []
            for i in range(len(df)):
                print("Processing Question",i+1)
                time.sleep(10)
                score,response = fusion_rag(query=df["query"][i])
                embed_score = calculate_bleu_rouge(response=response,groundtruth=df['truth'][i])
                print("\n******************************")
                print('query :- ', df['query'][i])
                print('truth :- ', df['truth'][i])
                print('\nLLM_response :- ', response) 
                print("*******\t Score \t***********") 
                print('relevance :- ', score['relevance']) 
                print('accuracy :- ', score['accuracy'])
                print('completeness :- ', score['completeness'])
                print('context_precision :- ', score['context_precision'])
                print('context_recall :- ', score['context_recall'])
                print('context_entities_recall :- ', score['context_entities_recall'])
                print('noise_sensitivity :- ', score['noise_sensitivity'])
                print('response_relevancy :- ', score['response_relevancy'])
                print('faithfulness :- ', score['faithfulness'])
                print('bleu :- ', embed_score['bleu'])
                print('rouge1 :- ', embed_score['rouge1'])
                print('rouge2 :- ', embed_score['rouge2'])
                print('rougeL :- ', embed_score['rougeL'])

                row = {
                'query': df['query'][i],
                'truth': df['truth'][i],
                '\nLLM_response': response,  
                'relevance': score['relevance'], 
                'accuracy': score['accuracy'],
                'completeness': score['completeness'],
                'context_precision': score['context_precision'],
                'context_recall': score['context_recall'],
                'context_entities_recall': score['context_entities_recall'],
                'noise_sensitivity': score['noise_sensitivity'],
                'response_relevancy': score['response_relevancy'],
                'faithfulness': score['faithfulness'],
                'bleu': embed_score['bleu'],  
                'rouge1': embed_score['rouge1'],
                'rouge2': embed_score['rouge2'],
                'rougeL': embed_score['rougeL']
                }
            
                data.append(row)
            fusion_rag_df = pd.DataFrame(data)
                                
            print("\nthis is a Fusion Rag Average Data Report")
            numeric_columns = fusion_rag_df.select_dtypes(include=['float64', 'int64'])
            average_values_simple = numeric_columns.mean()
            print(average_values_simple)
            
            
        case "corrective": 
            df = pd.read_csv("test.csv")
            data = []
            for i in range(len(df)):
                print("Processing Question",i+1)
                time.sleep(5)
                score,response = corrective_rag(query=df["query"][i])
                embed_score = calculate_bleu_rouge(response=response,groundtruth=df['truth'][i])
                print("\n******************************")
                print('query :- ', df['query'][i])
                print('truth :- ', df['truth'][i])
                print('\nLLM_response :- ', response) 
                print("*******\t Score \t***********") 
                print('relevance :- ', score['relevance']) 
                print('accuracy :- ', score['accuracy'])
                print('completeness :- ', score['completeness'])
                print('context_precision :- ', score['context_precision'])
                print('context_recall :- ', score['context_recall'])
                print('context_entities_recall :- ', score['context_entities_recall'])
                print('noise_sensitivity :- ', score['noise_sensitivity'])
                print('response_relevancy :- ', score['response_relevancy'])
                print('faithfulness :- ', score['faithfulness'])
                print('bleu :- ', embed_score['bleu'])
                print('rouge1 :- ', embed_score['rouge1'])
                print('rouge2 :- ', embed_score['rouge2'])
                print('rougeL :- ', embed_score['rougeL'])

                row = {
                'query': df['query'][i],
                'truth': df['truth'][i],
                '\nLLM_response': response,  
                'relevance': score['relevance'], 
                'accuracy': score['accuracy'],
                'completeness': score['completeness'],
                'context_precision': score['context_precision'],
                'context_recall': score['context_recall'],
                'context_entities_recall': score['context_entities_recall'],
                'noise_sensitivity': score['noise_sensitivity'],
                'response_relevancy': score['response_relevancy'],
                'faithfulness': score['faithfulness'],
                'bleu': embed_score['bleu'],  
                'rouge1': embed_score['rouge1'],
                'rouge2': embed_score['rouge2'],
                'rougeL': embed_score['rougeL']
                }
            
                data.append(row)
            corrective_rag_df = pd.DataFrame(data)
                                
            print("\nthis is a Corrective Rag Average Data Report")
            numeric_columns = corrective_rag_df.select_dtypes(include=['float64', 'int64'])
            average_values_simple = numeric_columns.mean()
            print(average_values_simple)

        case _:
            print("invalid input")

