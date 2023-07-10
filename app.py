import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template, text_template
# from langchain.llms import HuggingFaceHub
import youtube_gen
import pandas as pd
import os
import numpy as np
import csv
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
from streamlit_chat import message
import moviepy.editor
import imageio
import imageio.plugins.ffmpeg
import tempfile
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
openai.api_key = os.getenv("OPENAI_API_KEY")
user_secret = openai.api_key
data_transcription = []
data = []
audio_file_path = "audio.mp4"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# def convert_video_to_audio(video_path, audio_path):
#     # Load the video clip
#     video = VideoFileClip(video_path)

#     # Extract the audio from the video
#     audio = video.audio

#     # Save the audio as a new file
#     audio.write_audiofile(audio_path)

#     # Close the video file
#     video.close()

# # Provide the paths for the input video and output audio file
# local_audio_path = "audio.mp3"


# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)

# def handle_link_output(text):
#     st.write(text_template.replace(
#                     "{{MSG}}", text), unsafe_allow_html=True)
def main():
    load_dotenv()
    st.set_page_config(page_icon=":books:", page_title="Youtube Url Content",
                       )
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:        

        st.subheader("Provide the YouTube URL")
        url_link = st.text_input("Copy URL link here...")

        if st.button("Transcribe"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
            with st.spinner("Please Wait while process..."):
                url_raw_text = youtube_gen.get_url_text(url_link)
                raw_text = str(url_raw_text[0])

                #get the unique audio file name
                # st.write(url_raw_text[0][1])

                # audio_file = url_raw_text[0][1]
                
                audio_file_name = os.path.abspath(audio_file_path)
                st.warning(audio_file_name)

                if not os.path.exists(audio_file_name):
                    audio_file = open(audio_file_name, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/ogg')
                st.video(url_link) 

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                with open('transcription.txt', "w") as file:
                    file.write(raw_text)
                transcription = {
                    "title": "Youtube Video",
                    "transcription": raw_text
                }
                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv')
                # st.write(text_chunks)

                response = openai.Embedding.create(
                    input = raw_text,
                    model = 'text-embedding-ada-002'
                    )
                embeddings = response['data'][0]['embedding']
                meta = {
                    "text": "text",
                    "start": "Start",
                    "end": "end",
                    "embedding": embeddings
                }
                data.append(meta)
                print(embeddings)
                pd.DataFrame(data).to_csv('word_embeddings.csv') 
                # create vector store
                # vectorstore = get_vectorstore(text_chunks)

                # # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)

        st.subheader("Add your Local Files")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        # st.write(uploaded_file)
        if uploaded_file is not None:
            temp_dir = tempfile.TemporaryDirectory()
            
            # Create a temporary file path
            temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

            # Save the uploaded file to the temporary file path
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_file.read())

            # Display the path of the uploaded file
            st.success(f"File saved to: {temp_file_path} successfully")

            #convert to audio
            with st.spinner("Creating Audio File..."):
                video = moviepy.editor.VideoFileClip(temp_file_path)
                audio=video.audio
                if not os.path.exists("audio.mp3"):
                    audio.write_audiofile("audio.mp3")
                st.success("Audio file created successfully!")
        else:
            # Display a message if no file was uploaded
            st.info("Please upload a video file.")  
        if st.button("Transcribe", key="button1"):
            audio_file = open(temp_file_path, "rb")
            textt = openai.Audio.translate("whisper-1", audio_file)["text"]
            with open('transcription.txt', "w") as file:
                file.write(textt)
            transcription = {
                    "title": "Youtube Video",
                    "transcription": raw_text
                }
            data_transcription.append(transcription)
            pd.DataFrame(data_transcription).to_csv('transcription.csv')
            # st.write(text_chunks)

            response = openai.Embedding.create(
                input = raw_text,
                model = 'text-embedding-ada-002'
                )
            embeddings = response['data'][0]['embedding']
            meta = {
                "text": "text",
                "start": "Start",
                "end": "end",
                "embedding": embeddings
            }
            data.append(meta)
            print(embeddings)
            pd.DataFrame(data).to_csv('word_embeddings.csv') 


# Call the function to convert the video to audio

    tab1, tab2, tab3, tab4 = st.tabs(["Transcription", "Summary", "Embedding", "Chat with the Video"])
    with tab1: 
        st.header("Transcription:")
        if not os.path.exists(audio_file_path):
            audio_file = open(audio_file_path, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
        else :
            if not os.path.exists("audio.mp3"):
                audio_file_local = open("audio.mp3", 'rb')
                audio_bytes = audio_file_local.read()
                st.audio(audio_bytes, format='audio/ogg')
        if os.path.exists("transcription.txt"):
            with open("transcription.txt", "r") as file:
                transcription_text = file.read()
                st.write(transcription_text)
    with tab2:
        st.header("Summary:")

        def split_text(text):
            max_chunk_size = 2048
            chunks = []
            current_chunk = ""
            for sentence in text.split("."):
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += sentence + "."
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        def generate_summary(text):
            input_chunks = split_text(text)
            output_chunks = []
            for chunk in input_chunks:
                response = openai.Completion.create(
                    engine="davinci",
                    prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
                    temperature=0.5,
                    max_tokens=100,
                    n = 1,
                    stop=None
                )
                summary = response.choices[0].text.strip()
                output_chunks.append(summary)
            return " ".join(output_chunks)
        if os.path.exists("transcription.txt"):
            with open("transcription.txt", "r") as file:
                transcription_text = file.read()

        summary = generate_summary(transcription_text)
        st.write(summary)

    with tab3:
        st.header("Embedding:")
        if os.path.exists("word_embeddings.csv"):
            df = pd.read_csv('word_embeddings.csv')
            st.write(df)
    with tab4:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

        def get_text():
            if user_secret:
                st.header("Ask me something about the video:")
                input_text = st.text_input("You: ","", key="input")
                return input_text
        user_input = get_text()

        def get_embedding_text(api_key, prompt):
            openai.api_key = user_secret
            response = openai.Embedding.create(
                input= prompt.strip(),
                model="text-embedding-ada-002"
            )
            q_embedding = response['data'][0]['embedding']
            df=pd.read_csv('word_embeddings.csv', index_col=0)
            st.write(df['embedding'])
            df['embedding'] = df['embedding'].apply(eval).apply(np.array)

            df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
            returns = []
            
            # Sort by distance with 2 hints
            for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
                # Else add it to the text that is being returned
                returns.append(row["text"])

            # Return the context
            return "\n\n###\n\n".join(returns)

        def generate_response(api_key, prompt):
            one_shot_prompt = '''I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
            Q: What is human life expectancy in the United States?
            A: Human life expectancy in the United States is 78 years.
            Q: '''+prompt+'''
            A: '''
            completions = openai.Completion.create(
                engine = "text-davinci-003",
                prompt = one_shot_prompt,
                max_tokens = 1024,
                n = 1,
                stop=["Q:"],
                temperature=0.2,
            )
            message = completions.choices[0].text
            return message

        if user_input:
            text_embedding = get_embedding_text(user_secret, user_input)
            title = pd.read_csv('transcription.csv')['title']
            string_title = "\n\n###\n\n".join(title)
            user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
            # st.write(user_input_embedding)
            output = generate_response(user_secret, user_input_embedding)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        


                    
      
if __name__ == '__main__':
    main()