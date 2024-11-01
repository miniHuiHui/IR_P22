from flask import Flask, request, jsonify
from collections import defaultdict
import json
import re
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Define classes for Linked List Node and Linked List
class Node:
    def __init__(self, doc_id, tf_idf=0.0):
        self.doc_id = int(doc_id)  # Convert doc_id to numeric
        self.tf_idf = tf_idf
        self.next = None
        self.skip = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    def add(self, doc_id, tf_idf=0.0):
        new_node = Node(doc_id, tf_idf)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            prev = None
            # Insert in sorted order based on doc_id
            while current and current.doc_id < new_node.doc_id:
                prev = current
                current = current.next
            if prev is None:
                new_node.next = self.head
                self.head = new_node
            else:
                new_node.next = current
                prev.next = new_node
        self.length += 1

    def add_skip_pointers(self):
        if self.length <= 1:
            return
        skip_distance = round(math.sqrt(self.length))
        current = self.head
        count = 0
        prev_skip = None
        while current:
            if count % skip_distance == 0 and prev_skip:
                prev_skip.skip = current
            if count % skip_distance == 0:
                prev_skip = current
            current = current.next
            count += 1

    def show(self):
        current = self.head
        while current:
            current_next_id = current.next.doc_id if current.next else None
            skip_doc_id = current.skip.doc_id if current.skip else None
            print(f"Doc ID: {current.doc_id}, TF-IDF: {current.tf_idf}, Next: {current_next_id}, Skip to: {skip_doc_id}")
            current = current.next

    def get_postings_list(self):
        postings = []
        current = self.head
        while current:
            postings.append(current.doc_id)
            current = current.next
        return postings

    def get_skip_postings_list(self):
        skip_postings = []
        current = self.head
        while current:
            skip_postings.append(current.doc_id)
            current = current.skip
        return skip_postings

# Preprocessing steps
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and keep only alphanumeric and whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove excess whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize by whitespace
    tokens = text.split(' ')
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Perform stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Create inverted index
inverted_index = defaultdict(LinkedList)
document_lengths = defaultdict(int)
documents = {}  # Storing documents by their IDs
total_documents = 0

# Read input corpus
def build_inverted_index(input_file):
    global total_documents
    with open(input_file, 'r') as file:
        for line in file:
            total_documents += 1
            doc_id, sentence = line.strip().split('\t')
            doc_id = int(doc_id)  # Convert doc_id to numeric
            documents[doc_id] = sentence
            tokens = preprocess(sentence)
            # Count the frequency of tokens in the document
            token_frequency = defaultdict(int)
            for token in tokens:
                token_frequency[token] += 1

            # Update the inverted index
            for token, freq in token_frequency.items():
                tf = freq / len(tokens)
                inverted_index[token].add(doc_id, tf_idf=tf)  # Set the correct TF value initially

    # Add skip pointers to each postings list
    for token in inverted_index:
        inverted_index[token].add_skip_pointers()

    # Calculate IDF and update tf-idf scores
    for token, linked_list in inverted_index.items():
        idf = total_documents / linked_list.length
        current = linked_list.head
        while current:
            tf = current.tf_idf
            tf_idf = tf * idf
            current.tf_idf = tf_idf
            current = current.next

# Boolean Query Processing - Part 2

def preprocess_query(query):
    return preprocess(query)

def get_postings_list(term):
    return inverted_index.get(term, LinkedList())

def get_skip_postings_list(term):
    return inverted_index.get(term, LinkedList().get_skip_postings_list)

def daat_and_query(query_terms):
    postings_lists = [get_postings_list(term).head for term in query_terms if term in inverted_index]
    if not postings_lists:
        return [], 0

    # Sort postings lists by their length to optimize comparisons
    postings_lists = sorted(postings_lists, key=lambda node: inverted_index[query_terms[postings_lists.index(node)]].length if node else float('inf'))

    result = []
    num_comparisons = 0
    while all(postings_lists):
        current_ids = [node.doc_id for node in postings_lists if node is not None]
        if len(set(current_ids)) == 1:
            result.append(current_ids[0])
            postings_lists = [node.next for node in postings_lists if node is not None]
        else:
            min_doc_id = min(current_ids)
            for i in range(len(postings_lists)):
                if postings_lists[i] and postings_lists[i].doc_id == min_doc_id:
                    postings_lists[i] = postings_lists[i].next
        num_comparisons += 1

    return result, num_comparisons


def daat_and_query_sorted_by_tfidf(query_terms):
    postings_lists = [get_postings_list(term).head for term in query_terms if term in inverted_index]
    if not postings_lists:
        return [], 0

    # Sort postings lists by their length to optimize comparisons
    postings_lists = sorted(postings_lists, key=lambda node: inverted_index[query_terms[postings_lists.index(node)]].length if node else float('inf'))

    result = []
    num_comparisons = 0
    tfidf_scores = {}

    while all(postings_lists):
        current_ids = [node.doc_id for node in postings_lists if node is not None]
        if len(set(current_ids)) == 1:
            doc_id = current_ids[0]
            result.append(doc_id)
            tfidf_scores[doc_id] = sum(node.tf_idf for node in postings_lists if node and node.doc_id == doc_id)
            postings_lists = [node.next for node in postings_lists if node is not None]
        else:
            min_doc_id = min(current_ids)
            for i in range(len(postings_lists)):
                if postings_lists[i] and postings_lists[i].doc_id == min_doc_id:
                    postings_lists[i] = postings_lists[i].next
        num_comparisons += 1

    # Sort result by tf-idf scores in descending order
    result_sorted_by_tfidf = sorted(result, key=lambda doc_id: tfidf_scores[doc_id], reverse=True)

    return result_sorted_by_tfidf, num_comparisons

def daat_and_query_with_skips(query_terms):
    # 获取每个查询词的 postings 列表头部节点
    postings_lists = [get_postings_list(term).head for term in query_terms if term in inverted_index]
    if not postings_lists:
        return [], 0

    postings_lists = sorted(postings_lists, key=lambda node: inverted_index[query_terms[postings_lists.index(node)]].length if node else float('inf'))
    result = []
    num_comparisons = 0

    while all(postings_lists):  # 确保所有列表的头部节点都非空
        # 提取当前的文档 ID 列表
        current_ids = [node.doc_id for node in postings_lists]
        
        # 检查所有节点是否指向相同的 doc_id
        if len(set(current_ids)) == 1:
            # 所有 postings 指向相同的 doc_id，记录结果并前进到下一个节点
            result.append(current_ids[0])
            postings_lists = [node.next for node in postings_lists if node is not None]
        else:
            # 找到最小的 doc_id
            min_doc_id = min(current_ids)
            max_doc_id = max(current_ids)
            for i in range(len(postings_lists)):
                if postings_lists[i] and postings_lists[i].doc_id == min_doc_id:
                    # 尝试使用 skip pointer 跳跃，如果跳跃目标在 min_doc_id 或之后
                    if postings_lists[i].skip and postings_lists[i].skip.doc_id <= max_doc_id:
                        postings_lists[i] = postings_lists[i].skip
                    else:
                        # 否则顺序前进
                        postings_lists[i] = postings_lists[i].next
        # 增加比较次数
        num_comparisons += 1
        #print(f"Current IDs: {current_ids}, Min Doc ID: {min_doc_id}, Comparisons: {num_comparisons}")

    return result, num_comparisons

def daat_and_query_with_skips_sorted_by_tfidf(query_terms):
    # 获取每个查询词的 postings 列表头部节点
    postings_lists = [get_postings_list(term).head for term in query_terms if term in inverted_index]
    if not postings_lists:
        return [], 0

    result = []
    num_comparisons = 0
    tfidf_scores = {}

    while all(postings_lists):  # 确保所有列表的头部节点都非空
        # 提取当前的文档 ID 列表
        current_ids = [node.doc_id for node in postings_lists]
        
        # 检查所有节点是否指向相同的 doc_id
        if len(set(current_ids)) == 1:
            # 所有 postings 指向相同的 doc_id，记录结果并前进到下一个节点
            doc_id = current_ids[0]
            result.append(doc_id)
            tfidf_scores[doc_id] = sum(node.tf_idf for node in postings_lists if node and node.doc_id == doc_id)
            postings_lists = [node.next for node in postings_lists if node is not None]
        else:
            # 找到最小的 doc_id
            min_doc_id = min(current_ids)
            max_doc_id = max(current_ids)
            for i in range(len(postings_lists)):
                if postings_lists[i] and postings_lists[i].doc_id == min_doc_id:
                    # 尝试使用 skip pointer 跳跃，如果跳跃目标在 min_doc_id 或之后
                    if postings_lists[i].skip and postings_lists[i].skip.doc_id <= max_doc_id:
                        postings_lists[i] = postings_lists[i].skip
                    else:
                        # 否则顺序前进
                        postings_lists[i] = postings_lists[i].next
        # 增加比较次数
        num_comparisons += 1
        #print(f"Current IDs: {current_ids}, Min Doc ID: {min_doc_id}, Comparisons: {num_comparisons}")
        
        # Sort result by tf-idf scores in descending order
    result = sorted(result, key=lambda doc_id: tfidf_scores[doc_id], reverse=True)

    return result, num_comparisons







# Get postings lists for query terms
def get_postings_lists_for_terms(query_terms):
    postings = {}
    for term in query_terms:
        postings_list = get_postings_list(term).get_postings_list()
        postings[term] = postings_list
    return postings



# Get skip postings lists for query terms
def get_skip_postings_lists_for_terms(query_terms):
    skip_postings = {}
    for term in query_terms:
        skip_postings_list = get_postings_list(term).get_skip_postings_list()
        skip_postings[term] = skip_postings_list if skip_postings_list else []
    return skip_postings



@app.route('/execute_query', methods=['POST'])
#@app.route('/execute_query')
def execute_query():
    # Get the list of queries from request payload
 #   if not request.is_json:
 #       return jsonify({"error": "Invalid content type, JSON expected"}), 400
    # Get the list of queries from request payload
 #   queries = request.get_json(silent=True).get('queries', [])
    queries = request.json["queries"]
    response = {
        "Response": {
            "daatAnd": {},
            "daatAndSkip": {},
            "daatAndTfIdf": {},
            "daatAndSkipTfIdf": {},
            "postingsList": {},
            "postingsListSkip": {}
        }
    }

    for query in queries:
        terms = preprocess(query)
        all_terms = []
        all_terms+=terms
        for term in all_terms:
            print(term)
        # Postings List Retrieval
            response['Response']['postingsList'][term] = get_postings_lists_for_terms(terms)[term]
            response['Response']['postingsListSkip'][term]  = get_skip_postings_lists_for_terms(terms)[term]
        
        
        # DAAT AND Queries
        daat_results, num_comparisons = daat_and_query(terms)
        response['Response']['daatAnd'][query] = {
            "num_comparisons": num_comparisons,
            "num_docs": len(daat_results),
            "results": daat_results
        }

        # DAAT AND with Skip Pointers
        daat_skip_results, num_comparisons_skip = daat_and_query_with_skips(terms)
        response['Response']['daatAndSkip'][query] = {
            "num_comparisons": num_comparisons_skip,
            "num_docs": len(daat_skip_results),
            "results": daat_skip_results
        }

        # Sorting results by tf-idf
        sorted_results = daat_and_query_sorted_by_tfidf(terms)[0]
        response['Response']['daatAndTfIdf'][query] = {
            "num_comparisons": num_comparisons,
            "num_docs": len(sorted_results),
            "results": sorted_results
        }

        sorted_skip_results = daat_and_query_with_skips_sorted_by_tfidf(terms)[0]
        response['Response']['daatAndSkipTfIdf'][query] = {
            "num_comparisons": num_comparisons_skip,
            "num_docs": len(sorted_skip_results),
            "results": sorted_skip_results
        }

    # Save response to a JSON file
    with open('query_results.json', 'w') as file:
        json.dump(response, file, indent=4)

    return jsonify(response)

if __name__ == '__main__':
    # Build the inverted index before starting the server
    input_file = 'input_corpus.txt'
    build_inverted_index(input_file)
    app.run(host="0.0.0.0", port=9999)
