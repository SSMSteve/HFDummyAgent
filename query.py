import argparse
import params
from pymongo import MongoClient
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
import os

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = params.db_name
COLLECTION_NAME = params.collection_name
EMBEDDED_COLLECTION_NAME = params.embedded_collection_name
ATLAS_VECTOR_SEARCH_INDEX_NAME = params.index_name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

if args.question is None:
    # query = "What mclRoot has the Lead Attorney Goldstein?"
    # query = "Attorney for MLC Case Number: 20-02-00009?"
    # query="what is the attorney name for 18-09-00002 18-09-00002MDXT_Brislin_Conrado v CLS Landscapetorney ?"
    # query="Who is the lead attorney for 18-12-00020 18-12-00020MXAP_Bihr_Mayberry v LACMTA?"
    query="What is the firm for 18-12-00020 18-12-00020MXAP_Bihr_Mayberry v LACMTA?"
    # query = "What is the MlcRoot that has the Lead lawyer Goldstein"
    # query = "What are the MlcRoot that has the consultant Tsuruda. Form result as a list."
    # query = "What is the MlcRoot that has CLS Landscape"
    # query = "Were is the CRF located for  18-09-00002?"
    # query = "Were is the consultant for  18-09-00002?"
    # query = "When was the CRF completed for  18-09-00002? Just give the value of the date in the format YYYY/MM/DD."

else:
    query = args.question

print("\nYour question:")
print("-------------")
print(query)

# Initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[DB_NAME][EMBEDDED_COLLECTION_NAME]

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-large")
# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection, 
    embedding=embeddings, 
    index_name =ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# perform a similarity search between the embedding of the query and the embeddings of the documents
# print("\nQuery Response:")
print("---------------")
docs = vectorStore.similarity_search_with_score(query, k=1)
# docs = vectorStore.max_marginal_relevance_search(query, K=1)
for res, score in docs:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")    
# exit()
# if len(docs) == 0:
#     print("No documents found")
#     exit()
    
# for res, score in docs:
#     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")    
# exit()   

# if len(docs) == 0:
#     print("No documents found")
#     exit()
    
# # print(docs[0].metadata['database'])
# print(docs[0].page_content)

# Contextual Compression
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)

print("\nAI Response:")
print("-----------")
compressed_docs = compression_retriever.invoke(query)
# print(compressed_docs[0].metadata['title'])
# print(compressed_docs[0].page_content)
for res in compressed_docs:
    print(res.page_content)   
