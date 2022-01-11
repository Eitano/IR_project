# Information Retrieval Project:
Building a search engine for English Wikipedia and run the engine on VM.

# Data:
- Entire Wikipedia dump in a shared Google Storage as we already got at wikidumps
- Inverted index for title – a pickle file and posting locations in bins 
- Inverted index for anchor text – a pickle file and posting locations in bins 
- Inverted index for body text – a pickle file and posting locations in bins 
- Inverted index for Okapi BM25 for text anchor and body all together -  a pickle file and posting locations in bins.
- Pageviews - a pickle file
English Wikipedia from the month of August 2021, which is more than 10.7 million viewed articles (wiki articles pointing to other wiki articles)
- Pagerank - a pickle file
PageRank for wiki articles using the anchor text we extracted from the MediaWiki markdown 
- Queries and a ranked list of up to 100 relevant results for them, split into train (30 queries results given to you in queries_train.json).

# Files:
## 1)  Search_frontend.py:
A Flask app for search engine frontend. It has six methods implement:

  - Search - using Okapi BM25 on anchor text, title, and body text of the articles.
  - Cosine similarity using tf-idf on the body text of articles.
  - Binary ranking using the title of articles.
  - Binary ranking using the anchor text.
  - Ranking by PageRank -  finding corresponding page rank score to a page id.
  - Ranking by article page views - finding corresponding page views score to a page id.

### Key Optimization:
To reduce the time of retrieving an information from a bin file using only one inverted index (BM25.index).
For that we implemented an hash function which insert the same words that appears at anchor text, title and body to the same bins. 

### Key Implementation:
Okapi BM25 score is calculated over title , anchor and text and when we are figure how much a page is relevant by it position on the return list we are summing the BM25 scores of the matching word at title anchor and body text.

The file contains all the code needed to run the engine in VM which includes reading in advance the pickle file for saving time in the searching for quires. 

<br>

## 2)  Score.py:
  - A class contains the logical function of retrieving the candidate page match to the corresponding query to search for the above class.
     
<br>

## 3) InvertedIndex.py: 
  - An InvertedIndex module.

<br>

## 4) Preprocessing.ipynb:
Preprocessing file contains all the logic of information retrieval from Wikipedia, creating bins, posting list, inverted indexes, and as well other statistics calculations.
The code runs on GCP and works with MapReduce.

### Key implementation:
  - Tokenize a query from user – which use implemented clean text function and removing stop words and etc.
  - Finding candidates page from the corpus which contains the word in the query.
  - Call one of the functions at Search_frontend.py -  [Search, Cosine_ similarity, Binary title or anchor].
  - Retrieve the match id pages sorted by their match relevance score.

