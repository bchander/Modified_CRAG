# Modified_CRAG
Human-Feedback based Modified Corrective RAG

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
