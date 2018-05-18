# archi.codes

### TOC

1. [About](#About)
2. [Procedure](#Procedure)
3. [Technical Stack](#Technical_Stack)

### About
***NLP and knowledge graphs for AEC using spaCy***

archi.codes is an application that can help architects, engineers, and contractors quickly navigate through multiple building code documents and find the relevant information for their design and work. archi.codes uses the spaCy NLP model to analyze the text from building code documents and generates knowledge graphs from this information. Knowledge graphs will help to make automated compliance checking more possible and will lead to higher efficiency in the construction industry


### Procedure
1. Collect text data and split provisions into rows.
2. Run the spaCy NLP model on the text and extract keywords.
3. Construct knowledge graph from dependency parsed data.

### Technical Stack
![Tech Stack](slides/images/tech-stack.png)
