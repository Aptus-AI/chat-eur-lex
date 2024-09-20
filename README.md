# Chat-EUR-Lex Project and Prototype

This is the repository of the Chat-EUR-Lex project funded by [NGI Search](https://ngi-search-2nd-open-call.fundingbox.com/). The project aim is to improve the accessibility of EU laws, thus democratizing the availability of legal information for companies, lawyers, researchers and citizens. Navigating the complex legal terrain of [EUR-Lex](https://eur-lex.europa.eu/homepage.html) is not easy for citizens and companies, as the existing system requires a deep knowledge of legal jargon and categorizations. In this context, EUR-Lex is often difficult to use by non-experts.

In Chat-EUR-Lex we want to develop a trustworthy chatbot interface to access EU legal documents, providing simplified explanations, and fostering users' understanding of EU law.

The main technology underlying Chat-EUR-Lex prototype is Retrieval Augmented Generation (RAG): 
- [X] The legal documents are indexed in a vector DB; the user query is transformed into a semantic vector; the retriever looks into the vector DB for similar contents.
- [X] An LLM gets both the user query and the context (inserted in a prompt template) and generates the response to the user. Therefore, the generated answer refers to the portions of legal documents retrieved, and LLM hallucinations are reduced.
      
User query examples: "Explains in a simple way what the right to be forgotten is as defined in the GDPR"; "In terms of organic production, what does in-conversion production unit mean?"; "Summarize and simplify Directive (EU) 2023/2225 of the European Parliament and of the Council on consumer credit agreements".

Duration: from September 2023 to September 2024.

Chat-EUR-Lex project is realized by the [Institute of Legal Informatics and Judicial Systems (IGSG-CNR)](https://www.igsg.cnr.it/en/) and the [Aptus.AI](https://www.aptus.ai/) startup.

The project is funded by the European Union within the framework of the NGI Search project under grant agreement No 101069364 (NGI Search 2nd open call 2023). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Commission. 


## Roadmap

- [X] **Requirements and use cases definition**. In this initial phase, the project team identifies the specific requirements for the Chat-EUR-Lex system. This includes understanding the target audience's needs, such as citizens, lawyers, researchers, and companies. The team also defines the use cases, outlining how the AI-powered interface will assist users in navigating and understanding EU laws.
- [X]  **UX Research and Survey**. We collect data and information from a sample of potential users through a questionnaire, both in Italian and in English. Objective of the survey: understand the needs of people using digital legal resources, and their level of satisfaction; identify users needs and desires regarding chatbot interaction; know the fears related to the use of generative AI. User experience (UX) research involves studying how users interact with the current EUR-Lex system and identifying pain points and challenges.
- [X]  **Data collection and processing**. To power the AI system, a substantial amount of legal data from EUR-Lex is collected and processed (Italian and English languages).
- [X] **UX Validation and Wireframing**. Based on the research, the team creates a prototype of the Chat-EUR-Lex interface, mapping out the layout and design elements to ensure a user-friendly experience. The wireframes and prototypes are tested with potential users to gather feedback and validate the user experience. This step helps ensure that the interface meets users' needs and expectations.
- [X] **Semantic search engine setup**. Setting up a semantic search engine is a critical step. Semantic search allows users to find relevant legal information even if they don't use precise legal terminology. This involves using natural language processing (NLP) techniques to improve search accuracy.
- [X] **Semantic search engine tuning and validation**. The search engine is fine-tuned and validated to ensure that it effectively retrieves relevant legal documents and explanations. This tuning process may involve iterative adjustments based on experts' feedback and testing.
- [X] **RAG-based Chat system development**. The project team develops the core AI chat system using Retrieval Augmented Generation (RAG) techniques. RAG combines retrieval-based methods with generative language models to provide accurate and contextually relevant responses to user queries.
- [X] **UI Development**. While the chat system is being developed, the user interface (UI) is designed and implemented. The UI should be intuitive and easy to use, providing access to the AI-powered chat system.
- [ ] **First version release**. The initial version of Chat-EUR-Lex is released to a selected group of users. This version should provide basic functionality and serve as a starting point for further improvements.
- [ ] **Feedback collection and tuning**. After the first release, user feedback is actively collected and analyzed. This feedback is used to identify areas for improvement and fine-tune both the RAG and chat system, and the user interface. This iterative process continues to enhance the system's effectiveness and user satisfaction.

The prototype GUI is accessible on Hugging Face Spaces to authorized users: [https://huggingface.co/spaces/AptusAI/Chat-EUR-Lex](https://huggingface.co/spaces/AptusAI/Chat-EUR-Lex).

Scientific Publication: Manola Cherubini, Francesco Romano, Andrea Bolioli, Lorenzo De Mattei, Mattia Sangermano, “Improving the accessibility of EU laws: the Chat-EUR-Lex project”, ITAL-IA 2024 IV CINI national conference on artificial intelligence, Naples 29 may 2024
