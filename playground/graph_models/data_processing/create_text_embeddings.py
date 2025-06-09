from typing import List

import openai
import argparse
import numpy as np
import torch
import json
import os
import tiktoken
import tqdm
import sys

import spacy
import en_core_web_lg
nlp = spacy.load("en_core_web_lg")

sys.path.insert(0, '/home/klrshak/work/VisionLang/whereami-text2sgm/playground')
from graph_models.src.utils import load_text_dataset

# -------------------------------------------------------------------------------------------
# CHANGED to implement clip embeddings
from transformers import CLIPTokenizer, CLIPModel

# CHANGED to implement clip embeddings: initialize CLIP tokenizer & model
_clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
_clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)

# -------------------------------------------------------------------------------------------

# Load OpenAI API key
# with open(os.path.join(os.path.dirname(__file__), '/home/julia/Documents/h_coarse_loc/data/openai/openai_api_key.txt'), 'r') as f:
#     api_key = f.read().strip()
api_key = "sdauhdsihuais"
openai.api_key = api_key

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_list_of_strings(list_of_strings: List[str], encoding_name: str) -> int:
    """Returns the number of tokens in a list of text strings."""
    num_tokens = 0
    for string in list_of_strings:
        num_tokens += num_tokens_from_string(string, encoding_name)
    return num_tokens

def num_tokens_from_dict(dict_of_texts: dict, encoding_name: str) -> int:
    """Returns the number of tokens in a dictionary of text strings."""
    num_tokens = 0
    for scan_id in dict_of_texts:
        num_tokens += num_tokens_from_list_of_strings(dict_of_texts[scan_id], encoding_name)
    return 
    
def check_tokens():
    scan_ids, dict_of_texts = load_text_dataset()
    num_tokens = num_tokens_from_dict(dict_of_texts, 'cl100k_base')
    print(num_tokens)
    
# -------------------------------------------------------------------------------------------
# Function to create an embedding using OpenAI's API
# CHANGED to implement clip embeddings
def create_embedding_clip(text: str) -> torch.Tensor:
    """
    Returns a 512-dimensional CLIP text embedding for the given string.
    """
    inputs = _clip_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs   = _clip_model.get_text_features(**inputs)
        embedding = outputs.squeeze(0)
    return embedding

# CHANGED to implement clip embeddings (optional batch version)
def create_embeddings_clip_batch(texts: list[str]) -> torch.Tensor:
    """
    Returns an (N × 512) tensor of CLIP embeddings for a list of N strings.
    """
    inputs = _clip_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = _clip_model.get_text_features(**inputs)  # (N, 512)
    return outputs
# -------------------------------------------------------------------------------------------

def create_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding

def tokenize_text(filename):
    scan_ids, dict_of_texts = load_text_dataset(filename)
    dict_of_embeddings = {}
    for scan_id in tqdm.tqdm(dict_of_texts):
        dict_of_embeddings[scan_id] = []
        for text in dict_of_texts[scan_id]:
            embedding = create_embedding(text)
            dict_of_embeddings[scan_id].append(embedding)
    return dict_of_embeddings

def create_embedding_nlp(text):
    # spacy embedding
    doc = nlp(text)
    embedding = doc.vector
    assert(len(embedding) == 300)
    return embedding

def test_ada_embedding():
    worda = 'shelf'
    atta = ['brown']
    wordb = 'floor'
    attb = ['tiled']

    emba = create_embedding(worda)
    avg_atta = np.mean([create_embedding(att) for att in atta], axis=0)
    embb = create_embedding(wordb)
    avg_attb = np.mean([create_embedding(att) for att in attb], axis=0)

    print(f'cosine ab word only: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')

    emba = np.add(emba, avg_atta)
    embb = np.add(embb, avg_attb)

    emba_weighted_sum = np.add(emba, 0.2*avg_atta)
    embb_weighted_sum = np.add(embb, 0.2*avg_attb)

    # cosine similarity
    print(f'cosine ab: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')
    print(f'cosine ab weighted sum: {np.dot(emba_weighted_sum, embb_weighted_sum) / (np.linalg.norm(emba_weighted_sum) * np.linalg.norm(embb_weighted_sum))}')

def test_nlp_embedding():
    worda = 'shelf'
    wordb = 'bookshelf'
    wordc = 'yellow'
    wordd = 'Jacket'
    emba = create_embedding_nlp(worda)
    embb = np.add(create_embedding_nlp(wordb), create_embedding_nlp(wordc))
    embd = create_embedding_nlp(wordd)
    # cosine similarity
    print(f'cosine ab: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')
    # print(f'cosine ac: {np.dot(emba, embc) / (np.linalg.norm(emba) * np.linalg.norm(embc))}')
    print(f'cosine ad: {np.dot(emba, embd) / (np.linalg.norm(emba) * np.linalg.norm(embd))}')
    # print(f'cosine bc: {np.dot(embb, embc) / (np.linalg.norm(embb) * np.linalg.norm(embc))}')
    print(f'cosine bd: {np.dot(embb, embd) / (np.linalg.norm(embb) * np.linalg.norm(embd))}')
    # print(f'cosine cd: {np.dot(embc, embd) / (np.linalg.norm(embc) * np.linalg.norm(embd))}')

if __name__ == '__main__':
    test_ada_embedding()
    exit()

    ## check_tokens()
    ############################### Create embeddings for text dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    embeddings = tokenize_text(args.filename)

    # save
    with open('../scripts/hugging_face/scanscribe_2_embeddings.json', 'w') as fp:
        json.dump(embeddings, fp)
    
    # dict_of_embeddings = tokenize_text(args.filename)
    # torch.save(dict_of_embeddings, 'human+GPT_cleaned_text_embedding_ada_002.pt')

# ################################ Take user input to create a smaller dataset
    # scans_ids, dict_of_texts = load_text_dataset(args.filename)
    # dict_selection = {}
    # for key in dict_of_texts:
    #     # Go through all the examples in each dict_of_texts[key]
    #     user_input = input("Exit overall?")
    #     if user_input == 'exit':
    #         break
    #     else:
    #         pass
    #     for text in dict_of_texts[key]:
    #         print("Text: ", text)
        
    #         user_input = input("Add to dict_selection? (y/n): ")
    #         if user_input == 'y':
    #             if key not in dict_selection:
    #                 dict_selection[key] = []
    #             dict_selection[key].append(text)
    #         elif user_input == 'q':
    #             break
    #         else:
    #             continue

    # # Save dict_selection as a json in the ../scripts/hugging_face/ folder
    # with open('../scripts/hugging_face/scanscribe_2.json', 'w') as fp:
    #     json.dump(dict_selection, fp)
# ################################ Take user input to create a smaller dataset