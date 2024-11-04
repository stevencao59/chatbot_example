import json
import asyncio

from chatbot import ChatManager

default_message = "Tell me about yourself"

async def main(event: dict):
    manager = ChatManager()
    message: str = event.get('message', default_message)
    if 'movie' in message.lower():
        return await manager.handle_movie_message(message, event['viewerId'])
    return await manager.handle_general_message(message)

def lambda_handler(event: dict, context):
    result = asyncio.run(main(event))

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
