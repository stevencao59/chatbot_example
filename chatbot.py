import boto3
from boto3.dynamodb.conditions import Attr
import asyncio
from langchain_openai import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory.buffer import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from prompts import app_assistant_prompt

model = 'gpt-3.5-turbo-instruct'
viewer_table_name = 'viewers'
movie_table_name = 'movies'

class ChatManager(object):
    db_client = None

    def __init__(self):
        self.db_client = boto3.client('dynamodb')

    async def handle_general_message(self, message: str):
        llm = OpenAI(model=model, temperature=0)
        conversation_memory = ConversationBufferMemory(memory_key="chat_history", max_len=200, return_messages=True)
        llm_chain = LLMChain(llm=llm, prompt=app_assistant_prompt, memory=conversation_memory)
    
        viewer_message = message.lower()
        response = await llm_chain.acall(viewer_message)

        response_key = "output" if "output" in response else "text"
        return response.get(response_key, "")

    def get_movies(self, viewer_id):
        viewer_info = self.db_client.get_item(
            TableName=viewer_table_name,
            Key={'viewerId':{'S': viewer_id}}
        )
        associated_movie_ids = [a['S'] for a in viewer_info['Item']['movies']['L']]

        dynamodb = boto3.resource('dynamodb')
        movie_table = dynamodb.Table(movie_table_name)
        associated_movies = movie_table.scan(
            FilterExpression=Attr('movieId').is_in(associated_movie_ids)
        )
        return associated_movies

    def handle_movie_message(self, message, viewer_id):
        associated_movies = self.get_movies(viewer_id=viewer_id)

        # Convert movies into string array
        detail_list = [{**{'movieId': a['movieId']}, **a['movieDetails']} for a in associated_movies['Items']]
        movies = [f"movie: {b['movieId']} Name: {b['name']}, Year: {b['year']}, Director: {b['director']}, Actors: {b['actors']}" for i, b in enumerate(detail_list)]
        movies_str = '\n'.join(movies)
        movies_str = f'We have {len(movies)} movies in total. Each movie details are as below\n{movies_str}'

        # Embedding AI
        embeddings = OpenAIEmbeddings()

        # Text splitting into tokens
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        pages = text_splitter.split_text(movies_str)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents(pages)

        # Save in a vector store
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Get result from QA chain
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.7),
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True
        )

        response = chain({'query': message})
        return response.get('result', "")

async def main():
    manager = ChatManager()
    # result = await manager.handle_general_message("What's this app for?")
    # result = manager.handle_movie_message('Do we have a movie with address from NY?', "000210.c322a90eba814e5c88edb032b9afbc39.1533")
    # result = manager.handle_movie_message('How may movies we have?', "000210.c322a90eba814e5c88edb032b9afbc39.1533")
    # result = manager.handle_movie_message('Print out all movie id', "000210.c322a90eba814e5c88edb032b9afbc39.1533")
    # result = manager.handle_movie_message('Print out all movies with zip code 07310', "000210.c322a90eba814e5c88edb032b9afbc39.1533")
    result = manager.handle_movie_message('Provide me with some movie information', "000210.c322a90eba814e5c88edb032b9afbc39.1533")
    print(result)

if __name__ == "__main__":
    # s3 = boto3.resource('s3')

    # # Print out bucket names
    # for bucket in s3.buckets.all():
    #     print(bucket.name)

    asyncio.run(main())