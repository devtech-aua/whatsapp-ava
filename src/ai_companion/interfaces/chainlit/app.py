from io import BytesIO

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph.obenan_graph import obenan_graph
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings

# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", 1)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and images"""
    msg = cl.Message(content="")

    # Process any attached images
    content = message.content
    if message.elements:
        for elem in message.elements:
            if isinstance(elem, cl.Image):
                # Read image file content
                with open(elem.path, "rb") as f:
                    image_bytes = f.read()

                # Analyze image and add to message content
                try:
                    # Use global ImageToText instance
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                except Exception as e:
                    cl.logger.warning(f"Failed to analyze image: {e}")

    # Process through graph with enriched message content
    thread_id = cl.user_session.get("thread_id")

    async with cl.Step(type="run"):
        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            # Compile the graph with a checkpointer instead of using with_checkpointer
            graph = obenan_graph.compile(checkpointer=short_term_memory)
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=content)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                # Fix the condition to properly detect agent nodes
                if isinstance(chunk[0], AIMessageChunk) and chunk[1]["langgraph_node"].startswith("obi"):
                    await msg.stream_token(chunk[0].content)

            output_state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})

    # Debug: Log the output for troubleshooting
    cl.logger.info(f"Workflow: {output_state.values.get('workflow')}")
    cl.logger.info(f"Response: {output_state.values.get('messages', [])[-1].content if output_state.values.get('messages') else 'No message'}")

    if output_state.values.get("workflow") == "audio":
        response = output_state.values["messages"][-1].content
        audio_buffer = output_state.values["audio_buffer"]
        output_audio_el = cl.Audio(
            name="Audio",
            auto_play=True,
            mime="audio/mpeg3",
            content=audio_buffer,
        )
        await cl.Message(content=response, elements=[output_audio_el]).send()
    elif output_state.values.get("workflow") == "image" or output_state.values.get("workflow") == "vision":
        response = output_state.values["messages"][-1].content
        image_path = output_state.values.get("image_path")
        if image_path:
            image = cl.Image(path=image_path, display="inline")
            await cl.Message(content=response, elements=[image]).send()
        else:
            await msg.send()
    else:
        await msg.send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handle incoming audio chunks"""
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements):
    """Process completed audio input"""
    # Get audio data
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()

    # Show user's audio message
    input_audio_el = cl.Audio(mime="audio/mpeg3", content=audio_data)
    await cl.Message(author="You", content="", elements=[input_audio_el, *elements]).send()

    # Use global SpeechToText instance
    transcription = await speech_to_text.transcribe(audio_data)

    thread_id = cl.user_session.get("thread_id")

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        # Compile the graph with a checkpointer
        graph = obenan_graph.compile(checkpointer=short_term_memory)
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=transcription)]},
            {"configurable": {"thread_id": thread_id}},
        )

    # Use global TextToSpeech instance
    audio_buffer = await text_to_speech.synthesize(output_state["messages"][-1].content)

    output_audio_el = cl.Audio(
        name="Audio",
        auto_play=True,
        mime="audio/mpeg3",
        content=audio_buffer,
    )
    await cl.Message(content=output_state["messages"][-1].content, elements=[output_audio_el]).send()
