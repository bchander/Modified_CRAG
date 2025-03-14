'''
Project Name: Human-Feedback based Modified Corrective RAG

Project Description & Steps: 
Extraction of specifications from datasheets using Modified CRAG such that
1. The Model first tries to extract as much specifications from the user input datasheet using Basic RAG approach
2. The application allows users to give feedback about the generated response of the basic RAG. The users can select one of the three options: 'Correct', 'Incorrect', 'Ambiguous'
3. In this modified version of CRAG, I didn't use 'Retrieval Evaluator' as a grading mechanism to determine whether the fetched response is correct, incorrect or ambiguous. But instead I modified the approach to incliude a human feedback to take the agentic decision.
4. The steps for the user selections are given below:
    a. If the user selects 'Correct', the application will stick to the generated response.
    b. If the user selects 'Incorrect', the application will use rewrited query and 1) Extract specs based on rewritted query b) Identify the missing specs ('N/A') and use GPT's knowledge or websearch to fetch it .
    c. If the user selects 'Incomplete', the application will call an additional function (may be ask the GPT itself or use websearch) to fetch the missing information from the internet/knowledge base.
5. If the case is b and c in step 4, the application will further display the additional/regenerated information.


Typical Steps in Corrective RAG (CRAG):
For the user query, the Retriever retrives relevant documents.
The CRAG checks if the retrieved documents are relevant or not using a grader called 'Retrieval Evaluator'.
The 'RE' categorizes the generated responses as either 'coorect', 'incorrect' or 'ambiguous'.
    a. If the response is 'correct', the application proceeds to generation.
        Before generation, it performs knowledge refinement. This partitions the document into "knowledge strips"
        It grades each strip, and filters our irrelevant ones and only retains the relevant ones
    b. If the response is 'Incorrect', the application uses the 'rewrited query' and also do a websearch to generate some response.
    c. If the response is 'Ambiguous', the application does knowledge refinement step as well as do additional step of rewritting query and doing a web search. 

Reference for study: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/

Created by: Bhanu
Modified on: 7th Feb 2025
'''

#........Importing Required Libraries........#
from io import BytesIO
import json
import pandas as pd
import fitz
import streamlit as st
from streamlit_chat import message
from typing_extensions import TypedDict 
import openai
from openai import OpenAI
import numpy as np
from IPython.display import Image, display

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

#......Setting the environment variables manually......#
openai.api_key = "YOUR_API_KEY" 
gpt_model = 'MODEL_NAME'  
embedding_model = "EMBEDDING_MOEL"
client = OpenAI(api_key = "YOUR_API_KEY")

#....Defining the Prompt.....#
input_prompt = """Identify the specifications from the provided document context and extract the values for the identified specifications from the context. 
                      Provide the extracted specifications from the provided context in a structured JSON format. Each specification should have an value entry. 
                      The values should also include the associated units if available. If a specification is missing, include 'N/A' for that entry.
                      The first two entires of the specification should be 'company' and 'product/Model Number'. Extract as many specifications as possible from the provided contexts.
                      If the fetched values has any special characters like backslash etc., convert the values such that it will not create any issues while parsing the JSON. 
                      If the extracted value for any specification is in nested format, convert them into a list of key value pair and stick to the required format of JSON. Don't output in nested json format. Don't use same keys again and again, but instead merge similar key information into a list of strings.
                      Format your response as: 
                      {
                        "company": ["Value from context1"],
                        "product/Model Number": ["Value from context1"],
                        "Specification 3": ["Value from context1"], 
                        "Specification 4": ["Value from context1"], 
                        ... 
                      }
                      
                      """ 

#.......Function to extract text from PDF file......#
def extract_text_from_pdf(file) -> str:
    pdf_reader = fitz.open(stream=file.getvalue(), filetype="pdf")
    full_text = ""
    page_texts = []
    # for page_num in range(pdf_reader.getNumPages()):
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        text = page.get_text("text") 
        blocks = page.get_text("blocks")

        processed_blocks = []
        for b in blocks:
            block_text = b[4].strip()
            if block_text:
                if ":" in block_text:
                    processed_blocks.append(block_text)
                else:
                    processed_blocks.append(block_text.replace("\n", " "))
        processed_text = "\n".join(processed_blocks)
        full_text += processed_text + "\n\n"
        page_texts.append(processed_text)
        # st.session_state.pdf_text = page_texts
    return page_texts


#......Function to extract specs from the input datasheet text......#
def extract_specs(pdf1_text: str, input_prompt: str) -> str:
    text1 = pdf1_text
    # text2 = pdf2_text
    # .....Generating the responses using the input contexts......#
    response = client.chat.completions.create(
        model=gpt_model, 
        messages = [ {"role": "assistant", "content": input_prompt
                      },
                      {"role": "system", "content": f"context1: {text1}"},
                    #   {"role": "user", "content": query_text}
                      ], #prompt=prompt_template,
        max_tokens=1200,
        temperature = 0.1,
    )
    specifications = response.choices[0].message.content
    return specifications

#.....Function to rewrite the input prompt......#
def rewrite_query(input_prompt: str) -> str:
    # .....Generating the responses using the input prompt......#
    response = client.chat.completions.create(
        model=gpt_model, 
        messages = [ {"role": "assistant", "content": """You are a question re-writer that converts an input question to a better version that is optimized to fetch accurate answer from the input document. 
                      Look at the input and try to reason about the underlying semantic intent / meaning. Provide the response in a string format enclosed in thrible quotes.
                      
                      input question: {input_prompt}
                      """
                      },
                    #   {"role": "system", "content": f"context1: {text1}"},
                    #   {"role": "user", "content": query_text}
                      ], #prompt=prompt_template,
        max_tokens=600,
        temperature = 0.1,
    )
    rewrited_prompt = response.choices[0].message.content
    return rewrited_prompt

#..........Fetching additional information that seems to be missing from datasheet..........#
def fetch_additional_info(specifications_list: list) -> str:
    
    #.....Defining the specifications to be extracted from the context.....#
    company_name = specifications_list[0]
    product_model = specifications_list[1]
    missing_specs = specifications_list[2:]

    # .....Generating the responses for the missing specification......#
    response = client.chat.completions.create(
        model=gpt_model, 
        messages = [ {"role": "assistant", "content": """You are an expert in identified specifications when provided with Company Name and Product Model Number. Given a list of specifications to be extracted, you are required to use the given company name and model number and fetch the values for all the provided specifications. Inputs are:
                      Company Name: {company_name}
                      Model Number: {product_model}
                      and the list of specifications to be extracted: {missing_specs}
                      
                      For all the specifications to be fetched provide the identified response in a structured JSON format. Each specification should have an value entry. 
                      The values should also include the associated units if available. If a specification is missing, include 'N/A' for that entry. For all the non-missing specs include the text '(from internet)' along with the values.
                      If the fetched values has any special characters, convert the values such that it will not create any issues while parsing the JSON.

                      Format your response as: 
                      {
                        "Specification 1": ["Value (from internet)"], ..
                        "Specification 2": ["Value (from internet)"],
                        ... 
                      }
                      """
                      },
                      # {"role": "system", "content": f"context1: {text1}"},
                      # {"role": "user", "content": query_text}
                      ], #prompt=prompt_template,
        max_tokens=1200,
        temperature = 0.1,
    )
    specifications = response.choices[0].message.content
    return specifications

#..........Functions to handle different user selections..........#

#.....Function to handle situation when the user selects correct.....#
def handle_correct_action(state):
    return {"filtered_specs": state["specifications"], "additional_info": None}
    # return state

#.....Function to handle situation when the user selects incomplete.....#
def handle_incomplete_action(specs):
    specifications = json.loads(specs)
    keywords_list = ["Display", "Battery", "Processor", "Memory", "Storage", "Operating System", "Price", "Weight", "Connectivity", "Guarantee", "Graphics", "Battery Life", "Material", "Refresh Rate"]
    missed_keywords = ["company", "Model Number"]  # Initialize missed keywords with company and product/model number
    
    # Get the list of keys from the extracted specifications
    spec_keys = [key for key, value in specifications.items()]
    
    # Find the missed keywords
    for keyword in keywords_list:
        if keyword not in spec_keys:
            missed_keywords.append(keyword) 

    # Find keys with value 'N/A'
    keys_with_na = [key for key, value in specifications.items() if value == ["N/A"]]
    # filtered_specs = {key: value for key, value in specifications.items() if value != ["N/A"]}

    # Add missed keys and fetch information for the missed keys using the function
    all_missed_keys = missed_keywords + keys_with_na
    additional_info = fetch_additional_info(all_missed_keys)

    return additional_info
    # return state


#.....Function to handle situation when the user selects incorrect.....#
def handle_incorrect_action(pdf_text, input_prompt):

    # Re-Extracting specification using LLM    
    specifications = extract_specs(pdf_text, input_prompt)
    try:
        # Convert JSON string to a Python dictionary
        if isinstance(specifications, str):  # Ensure it's a string before parsing
            spec_type = type(specifications)
            specifications = json.loads(specifications)
    except:
        specifications = json.loads(specifications)
        st.error("error in loading spec as json")

    # specifications = json.loads(specifications)
    # Mentioning the list of keywords
    keywords_list = ["Display", "Battery", "Processor", "Memory", "Storage", "Operating System", "Price", "Weight", "Connectivity", "Guarantee", "Graphics", "Battery Life", "Material", "Refresh Rate"]
    missed_keywords = ["company", "Model Number"]  # Initialize missed keywords with company and product/model number
    
    # Get the list of keys from the extracted specifications
    if isinstance(specifications, dict):
        spec_keys = [key for key, value in specifications.items()]
    else:
        spec_keys = [key for key, value in specifications.items()]
    
    # Find the missed keywords
    for keyword in keywords_list:
        if keyword not in spec_keys:
            missed_keywords.append(keyword) 

    # Find keys with value 'N/A'
    keys_with_na = [key for key, value in specifications.items() if value == ["N/A"]]
    filtered_specs = {key: value for key, value in specifications.items() if value != ["N/A"]}

    # Concatenating all missed keys
    all_missed_keys = missed_keywords + keys_with_na
    additional_info = fetch_additional_info(all_missed_keys)

    return filtered_specs, additional_info
    # return state




###............Main Function.............###
def main():
    import time
    st.header("Specification extraction using Modified Corrective RAG")
    
    # Ensure session state variables exist
    if "specifications" not in st.session_state:
        st.session_state.specifications = None
        st.session_state.pdf_text = None
        
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None

    # Define the graph
    class State(TypedDict):
        uploaded_file_1: bytes
        pdf1_text: str
        specifications: str

    # Upload PDF files 
    uploaded_file_1 = st.file_uploader("Upload First PDF", type="pdf")
    
    submit=st.button("Submit")

    # Display uploaded files
    if submit and uploaded_file_1:
        with st.spinner("Fetching Specifications.....Please Wait"):
            
            #..........Create the graph..........#
            graph_builder = StateGraph(State)

            #..........Define the nodes..........#
            graph_builder.add_node("extract_text_doc1", lambda state: {"pdf1_text": extract_text_from_pdf(state["uploaded_file_1"])})
            # graph_builder.add_node("extract_text_doc2", lambda state: {"pdf2_text": extract_text_from_pdf(state["uploaded_file_2"])})
            graph_builder.add_node("extract_specs", lambda state: {"specifications": extract_specs(state["pdf1_text"], input_prompt)})
            
            # Define the edges that connects the nodes
            graph_builder.add_edge(START, "extract_text_doc1")
            graph_builder.add_edge("extract_text_doc1","extract_specs")
            graph_builder.add_edge("extract_specs", END)

            # Execute the graph
            graph = graph_builder.compile()

            # Assuming pdf1_file and pdf2_file are the uploaded PDF files in bytes
            input_state = {
                "uploaded_file_1": uploaded_file_1,
            }

            # Execute the graph
            result = graph.invoke(input_state)
            # Access the extracted specifications
            st.session_state.specifications = result["specifications"]
            st.session_state.pdf_text = result["pdf1_text"]

    if st.session_state.specifications:
        st.write("Specifications extracted from the provided datasheet")
        st.write(st.session_state.specifications)

        #.....Function to perform action ased on user feedback.....#
        def feedback_loop(button_pressed, specifications, pdf_text):
            with st.spinner("Processing your feedback.....Please Wait"):
                # Based on user feedback through button click for the generated response, take appropriate action
                if button_pressed == "Correct":
                    filtered_specs = specifications
                    additional_info = None
                elif button_pressed == "Incorrect":
                    # pdf_text = extract_text_from_pdf(uploaded_file_1)
                    filtered_specs, additional_info = handle_incorrect_action(pdf_text, input_prompt)
                    st.write("Query rewrited and done a fresh search. Fetched additional information from the internet")
                elif button_pressed == "Incomplete":
                    filtered_specs = specifications
                    additional_info = handle_incomplete_action(filtered_specs)
                    st.write("Additional information fetched from the internet")
                
                # Parse the JSON response
                if filtered_specs:
                    try:
                        if button_pressed == "Incorrect" or button_pressed == "Incomplete":
                            # if button_pressed == "Correct":
                            data2 = json.loads(additional_info)
                            df2 = pd.DataFrame(data2)

                            try:
                                data = json.loads(filtered_specs)
                                df1 = pd.DataFrame(data)
                            except:
                                try:
                                    df1 = pd.DataFrame([filtered_specs])
                                except:
                                    df1 = pd.DataFrame.from_dict(filtered_specs, orient='index')

                            df1 = df1.transpose().reset_index()
                            df2 = df2.transpose().reset_index()

                            # Concatenate the DataFrames
                            combined_df = pd.concat([df1, df2], ignore_index=True)
                            combined_df.columns = ['Specification', 'Extracted Information']
                        else:
                            try:
                                data = json.loads(filtered_specs)
                                combined_df = pd.DataFrame(data)
                            except:
                                try:
                                    combined_df = pd.DataFrame([filtered_specs])
                                except:
                                    combined_df = pd.DataFrame.from_dict(filtered_specs, orient='index')
                            combined_df = combined_df.transpose().reset_index()
                            combined_df.columns = ['Specification', 'Extracted Information']
                            # combined_df = pd.DataFrame(filtered_specs)                        

                        
                        st.write("JSON data loaded successfully.")

                        # Convert DataFrame to Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            combined_df.to_excel(writer, index=False, sheet_name='comparison_results')
                            writer.close()
                        excel_data = output.getvalue()
                        
                        # Provide download button
                        st.download_button(
                            label="Download data as Excel",
                            data=excel_data,
                            file_name='comparison_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )

                        #..........Displaying the generated Graph from LangGraph..........#
                        try:
                            # Visualize the graph
                            graph_image = Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))
                            st.write("Graph generated by LangGraph")

                            # Create a download button for the image
                            st.download_button(
                                label="Download Image",
                                data=graph_image,
                                file_name="graph.png",
                                mime="image/png"
                            )
                            display(graph_image)
                        except Exception:
                            # st.error(f"An error occurred while generating the graph image: {e}")
                            graph_image = None
                            
                        if graph_image:
                            st.image(display(graph_image), caption="Graph generated by LangGraph", use_column_width=True)
                        else:
                            st.error("Graph image could not be generated.")

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    print("spec_list is empty or None.")
        #.....END OF FUNCTION......#

        #.....Buttons to take user feedback......#
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Correct"):
                st.session_state.button_pressed = "Correct"
                feedback_loop(st.session_state.button_pressed, st.session_state.specifications, st.session_state.pdf_text)
        with col2:
            if st.button("Incorrect"):
                st.session_state.button_pressed = "Incorrect"
                feedback_loop(st.session_state.button_pressed, st.session_state.specifications, st.session_state.pdf_text)
        with col3:
            if st.button("Incomplete"):
                st.session_state.button_pressed = "Incomplete"
                feedback_loop(st.session_state.button_pressed, st.session_state.specifications, st.session_state.pdf_text)

if __name__=="__main__":
    main()