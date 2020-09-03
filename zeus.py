# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/BNkosi/Zeus/blob/master/Zeus.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Zeus.py
# %% [markdown]
# ## Contents
# 1. First installation
# 2. Imports
# 3. Data
# 4. Data cleaning and Preprocessing
# 5. Retriever
# 6. Reader
# 7. Finder
# 8. Prediction

# %%
# First instalation
get_ipython().system('pip install git+https://github.com/deepset-ai/haystack.git')
get_ipython().system('pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html')


# %%
# Make sure you have a GPU running
get_ipython().system('nvidia-smi')

# %% [markdown]
# ## Imports

# %%
# Minimum imports
from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.database.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

# %% [markdown]
# ## Load Data

# %%
def fetch_data_from_repo(doc_dir = "data5/website_data/", 
                         s3_url = "https://github.com/Thabo-5/Chatbot-scraper/raw/master/txt_files.zip",
                         doc_store=FAISSDocumentStore()):
    """
    Function to download data from s3 bucket/ github
    Parameters
    ----------
        doc_dir (str): path to destination folder
        s3_url (str): path to download zipped data
        doc_store (class): Haystack document store
    Returns
    -------
        document_store (object): Haystack document store object
    """
    document_store=doc_store
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    import os
    for filename in os.listdir(path=doc_dir):
        with open(os.path.join(doc_dir, filename), 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
            file.close()
        with open(os.path.join(doc_dir, filename), 'w', encoding='utf-8', errors='replace') as file:
            file.write(text)
            file.close()
    # Convert files to dicts
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(dicts)
    return document_store


# %%
document_store = fetch_data_from_repo()

# %% [markdown]
# ## Initialize Retriver, Reader and Finder

# %%
def initFinder():
    """
    Function to initiate retriever, reader and finder
    Parameters
    ----------
    Returns
    -------
        finder (object): Haystack finder
    """
    retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=False,
                                  embed_title=True,
                                  max_seq_len=256,
                                  batch_size=16,
                                  remove_sep_tok_from_untitled_passages=True)
    # Important: 
    # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation. 
    # While this can be a time consuming operation (depending on corpus size), it only needs to be done once. 
    # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
    document_store.update_embeddings(retriever)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    return Finder(reader, retriever)


# %%
finder = initFinder()


# %%
def getAnswers(retrieve=3, read=5, num_answers=1):
    while(True):
        query = input("You: ")
        if query == "bye":
            print("Goodbye!")
            break
        prediction = finder.get_answers(question=query, top_k_retriever=retrieve, top_k_reader=read)
        for i in range(0, num_answers):
            print(f"\nAnswer\t: {prediction['answers'][i]['answer']}")
            print(f"Context\t: {prediction['answers'][i]['context']}")
            print(f"Document name\t: {prediction['answers'][i]['meta']['name']}")
            print(f"Probability\t: {prediction['answers'][i]['probability']}\n\n")


# %%
getAnswers()


# %%
getAnswers(5,3,1)


# %%



