from langchain_core.documents import Document

doc = Document(
    page_content="This is your text of the document",
    metadata={
        "source": "example.txt",
        "pages": 2,
        "author": "Rakesh Anvekar"
    }
)

print(doc)
