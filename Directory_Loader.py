from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFLoader
dir_Loader_txt= DirectoryLoader("Data/",
                            glob="**/*.txt", ## patern to match
                            loader_cls=TextLoader,
                            loader_kwargs={'encoding':'utf-8'}                           
                            )

textDocuments= dir_Loader_txt.load()
print(textDocuments)

dir_Loader_pdf= DirectoryLoader("Data/",
                            glob="**/*.pdf", ## patern to match
                            loader_cls=PyPDFLoader,                         
                            )


pdfDocuments= dir_Loader_pdf.load()
print(pdfDocuments)