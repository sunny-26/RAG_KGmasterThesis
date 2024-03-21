Architecture
============

### Files naming conventions
------------------------
1. All the files for providing the functionality to do RAG should be stored in "enrich" folder.

2. Files should be named as the steps in RAG-approach
        * Noun
            * retrieval, augumentation...

## Rules for separation of functions
------------------------
1. Every function which is used to generate the data for the context should be stored in the corresponding file.
2. Every function which is used to derive the context should be stored in the corresponding file.
3. Every function which is used to create the context and generate response should be stored in the corresponding file.
4. The main flow of the approach should be stored in the seperate file.
5. The functions, which are not directly a part of RAG, should be stored in support_functions.py.





