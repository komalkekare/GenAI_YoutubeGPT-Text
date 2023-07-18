import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
import whisper
import whisperx
from pyannote.audio import Pipeline
import sys
import time
from pyAudioAnalysis import audioSegmentation as aS
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, text_template
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
from io import BufferedReader
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",device=torch.device("cpu"))
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
openai.api_key = os.environ.get("OPENAI_API_KEY")
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
user_secret = openai.api_key
data_transcription = []
data = []
# whisperoutput=[]
# whisperdata=[]
# DiarizeSpeaker=[]
audio = "audio.mp3"
# device = "cpu" 
# audio_file = "audio"
# batch_size = 16 # reduce if low on GPU mem
# compute_type = "float32"

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
        url_link = st.text_input("Copy URL link here...", placeholder="https://www.youtube.com/<Video_id>")

        if st.button("Start Analysis"):
            # if os.path.exists("word_embeddings.csv"):
            #     os.remove("word_embeddings.csv")
            os.remove("audio.mp3")

            with st.spinner("Please Wait while process..."):
                url_raw_text, audio_file_path = youtube_gen.get_url_text(url_link) 
                audio_file = os.path.abspath(audio_file_path)
                st.warning(audio_file)

                # audio = open(audio_file, 'rb')
                # if(os.path.exists(audio_file)):
                #     audio_file_name = open(audio_file, 'rb')
                #     audio_bytes = audio_file_name.read()
                #     st.audio(audio_bytes, format='audio/ogg')
                st.video(url_link) 

                # Whisper
                model = whisper.load_model("base")
                result = model.transcribe(audio_file, task='translate')
                # to_english = model.transcribe(audio_file, task='translate')
                # Transcription
                transcription = {
                    "title": "Youtube Video",
                    "transcription": result['text']
                }

                with open('transcription.txt', "w") as file:
                    file.write(result['text'])

                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 
                segments = result['segments']

                # Text-Embedding
                for segment in segments:
                    openai.api_key = user_secret
                    response = openai.Embedding.create(
                        input= segment["text"].strip(),
                        model="text-embedding-ada-002"
                    )
                    embeddings = response['data'][0]['embedding']
                    meta = {
                        "text": segment["text"].strip(),
                        "start": segment['start'],
                        "end": segment['end'],
                        "embedding": embeddings
                    }
                    data.append(meta)
                    time.sleep(20)
                pd.DataFrame(data).to_csv('word_embeddings.csv') 

                #Summary
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
                with open("Summary.txt", "w") as file:
                    file.write(summary)

                #Sentiment Analysis
                 # Read the file contents into a list of rows
                filename = "word_embeddings.csv"
                # Initialize an empty list to store rows
                rows = []
                data1=[]

                # Open the file in read mode with UTF-8 encoding
                with open(filename, "r", encoding="utf-8") as f:
                    # Create a CSV reader object
                    reader = csv.reader(f)
                    # Read the header row and store it separately
                    header = pd.read_csv('sentiment.csv', sep=',', names=['text','embedding'])
                    # Iterate over each row in the file and append it to the rows list
                    for row in reader:
                        rows.append(row)

                # Add a new header for the sentiment column
                header.append("Sentiment")

                # Loop over each row and perform sentiment analysis
                for row in rows:
                    # Extract the text to be analyzed from the Second column of the row
                    text = row[1]
                    # Create a prompt for the sentiment analysis API with the text
                    prompt = f"Please analyze the sentiment of the following text:{text}"
                    # Call the sentiment analysis API with the prompt
                    response = openai.Completion.create(
                        engine="text-davinci-002",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=128,  # Increase max_tokens to retrieve more than one token
                        n=1,
                        stop=None,
                        timeout=10,
                    )
                    # Extract the sentiment from the API response
                    sentiment = response.choices[0].text.strip().replace("The sentiment of the text is ", "").rstrip('.')
                    # Map the sentiment to a more concise label
                    if "Positive" in sentiment:
                        sentiment = "Positive"
                    elif "Negative" in sentiment:
                        sentiment = "Negative"
                    elif "Neutral" in sentiment:
                        sentiment = "Neutral"
                    # Append the sentiment to the row
                    row.append(sentiment)
                    # Print the text and its corresponding sentiment
                    # print(f"{text} -> {sentiment}")
                    # st.write(f"{text} -> {sentiment}")
                    meta = {
                        "text": row,
                        "Sentiment": sentiment
                    }
                    data1.append(meta)
                    print(data1)
                    pd.DataFrame(data1).to_csv('sentiment.csv') 

                    # Pause for 0.5 seconds to avoid hitting API rate limits
                    time.sleep(20)
            


                # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                #                     use_auth_token=hf_token)
            
                # # apply the pipeline to an audio file
                # diarization = pipeline(audio_file_name, num_speakers=2)

                # # dump the diarization output to disk using RTTM format
                # with open("audio.txt", "w") as rttm:
                #     diarization.write_rttm(rttm)

                
                # os.remove(audio_file)
                st.success('Analysis completed')

                # # Perform speaker diarization
                # result = aS.speaker_diarization(audio_file, num_speakers=2)

                # # Print the speaker segments
                # for segment in result:
                #     start_time, end_time, speaker_label = segment
                #     print(f"Speaker {speaker_label}: {start_time:.2f} - {end_time:.2f} seconds")
                #     st.write(f"Speaker {speaker_label}: {start_time:.2f} - {end_time:.2f} seconds")

                # st.write(str(audio_file))
                # raw_text = str(url_raw_text[0])

                #get the unique audio file name
                # st.write(url_raw_text[0][1])

                # audio_file = url_raw_text[0][1]
                
                # audio_file_name = os.path.abspath(audio_file)
                # st.warning(audio_file_name)

                # if(os.path.exists(audio_file_name)):
                #     audio_file = open(audio_file_name, 'rb')
                #     audio_bytes = audio_file.read()
                #     st.audio(audio_bytes, format='audio/ogg')
                # st.video(url_link) 

                # # get the text chunks
                # text_chunks = get_text_chunks(raw_text)

                # with open('transcription.txt', "w") as file:
                #     file.write(raw_text)
                # transcription = {
                #     "title": "Youtube Video",
                #     "transcription": raw_text
                # }
                # data_transcription.append(transcription)
                # pd.DataFrame(data_transcription).to_csv('transcription.csv')
                # # st.write(text_chunks)

                # response = openai.Embedding.create(
                #     input = raw_text,
                #     model = 'text-embedding-ada-002'
                #     )
                # embeddings = response['data'][0]['embedding']
                # meta = {
                #     "text": "text",
                #     "start": "Start",
                #     "end": "end",
                #     "embedding": embeddings
                # }
                # data.append(meta)
                # print(embeddings)
                # pd.DataFrame(data).to_csv('word_embeddings.csv') 
                # create vector store
                # vectorstore = get_vectorstore(text_chunks)

                # # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)
        
            # model = whisperx.load_model("large-v2", device, compute_type=compute_type)
            # audio = whisperx.load_audio(audio_file)
            # result = model.transcribe(audio, batch_size=batch_size)
            # segments = result['segments']
            # print(result["segments"]) # before alignment
            # st.write(result["segments"])

            # with open('whispertranscription.csv', "w") as file:
            #     for segment in segments:
            #         meta = {
            #             "text": segment["text"].strip(),
            #             "start": segment['start'],
            #             "end": segment['end']
            #         }
            #     whisperdata.append(meta)
            # pd.DataFrame(whisperdata).to_csv('timestamptranscription.csv') 

            # diarizeresult = diarize.transcribe_and_diarize(audio_file, hf_token,"base")
            # st.write(diarizeresult)

            # # 2. Align whisper output
            # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # print(result["segments"]) # after alignment
            # st.write(result["segments"])

            # for segment in segments:
            #     meta = {
            #         "text": segment["text"].strip(),
            #         "start": segment['start'],
            #         "end": segment['end']
            #     }
            #     whisperoutput.append(meta)
            # pd.DataFrame(whisperoutput).to_csv('whisperoutput.csv') 


            # # 3. Assign speaker labels
            # diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            # diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
            # diarization_result = diarization_pipeline(audio_file_name)
            # st.write(diarization_result)

            # aligned_segments = align_segments(
            #         transcript["segments"], transcript["language_code"], audio_file, device
            #     )
            # results_segments_w_speakers = assign_speakers(diarization_result, result)

            # add min/max number of speakers if known
            # diarize_segments = diarize_model(audio_file_name)
            # diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)

            # result = whisperx.assign_word_speakers(diarize_segments, result)
            # print(diarize_segments)
            # print(result["segments"]) # segments are now assigned speaker IDs
            # st.write(result["segments"])

            # for segment in segments:
            #     meta = {
            #         "text": segment["text"].strip(),
            #         "start": segment['start'],
            #         "end": segment['end']
            #     }
            #     DiarizeSpeaker.append(meta)
            # pd.DataFrame(DiarizeSpeaker).to_csv('DiarizeSpeaker.csv') 

        st.divider()

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

        if st.button("Start Analysis", key="button1"):
            audio_file = open(temp_file_path, "rb")
            # Whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file, task='translate')
            # to_english = model.transcribe(audio_file, task='translate')
            # Transcription
            transcription = {
                "title": "Youtube Video",
                "transcription": result['text']
            }

            with open('transcription.txt', "w") as file:
                file.write(result['text'])

            data_transcription.append(transcription)
            pd.DataFrame(data_transcription).to_csv('transcription.csv') 
            segments = result['segments']


            textt = openai.Audio.translate("whisper-1", audio_file)["text"]
            with open('transcription.txt', "w") as file:
                file.write(textt)
            transcription = {
                    "title": "Youtube Video",
                    "transcription": textt
                }
            data_transcription.append(transcription)
            pd.DataFrame(data_transcription).to_csv('transcription.csv')
            # st.write(text_chunks)

            response = openai.Embedding.create(
                input = textt,
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

            #Summary
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
            with open("Summary.txt", "w") as file:
                file.write(summary)

            #Sentiment Analysis
            # Read the file contents into a list of rows
            filename = "word_embeddings.csv"
            # Initialize an empty list to store rows
            rows = []
            data1=[]

            # Open the file in read mode with UTF-8 encoding
            with open(filename, "r", encoding="utf-8") as f:
                # Create a CSV reader object
                reader = csv.reader(f)
                # Read the header row and store it separately
                header = next(reader)
                # Iterate over each row in the file and append it to the rows list
                for row in reader:
                    rows.append(row)

            # Add a new header for the sentiment column
            header.append("Sentiment")

            # Loop over each row and perform sentiment analysis
            for row in rows:
                # Extract the text to be analyzed from the Second column of the row
                text = row[1]
                # Create a prompt for the sentiment analysis API with the text
                prompt = f"Please analyze the sentiment of the following text:{text}"
                # Call the sentiment analysis API with the prompt
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=128,  # Increase max_tokens to retrieve more than one token
                    n=1,
                    stop=None,
                    timeout=10,
                )
                # Extract the sentiment from the API response
                sentiment = response.choices[0].text.strip().replace("The sentiment of the text is ", "").rstrip('.')
                # Map the sentiment to a more concise label
                if "Positive" in sentiment:
                    sentiment = "Positive"
                elif "Negative" in sentiment:
                    sentiment = "Negative"
                elif "Neutral" in sentiment:
                    sentiment = "Neutral"
                # Append the sentiment to the row
                row.append(sentiment)
                # Print the text and its corresponding sentiment
                # print(f"{text} -> {sentiment}")
                st.write(f"{text} -> {sentiment}")
                meta = {
                    "text": row,
                    "Sentiment": sentiment
                }
                data1.append(meta)
                print(data1)
                pd.DataFrame(data1).to_csv('sentiment.csv') 
                # Pause for 0.5 seconds to avoid hitting API rate limits
                time.sleep(20)
            
         


# Call the function to convert the video to audio

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Introduction","Transcription", "Summary", "Embedding", "Chat with the Video", "Sentiment Analysis", "Speaker Diarization"])
    
    with tab1:
        # st.markdown("### How does it work?")
        # st.markdown('Read the article to know how it works: [Medium Article]("https://medium.com/@dan.avila7/youtube-gpt-start-a-chat-with-a-video-efe92a499e60")')
        st.write("Youtube GPT was written with the following tools:")
        st.markdown("#### Code GPT")
        st.write("All code was written with the help of Code GPT. Visit [codegpt.co]('https://codegpt.co') to get the extension.")
        st.markdown("#### Streamlit")
        st.write("The design was written with [Streamlit]('https://streamlit.io/').")
        st.markdown("#### Whisper")
        st.write("Video transcription is done by [OpenAI Whisper]('https://openai.com/blog/whisper/').")
        st.markdown("#### Embedding")
        st.write('[Embedding]("https://platform.openai.com/docs/guides/embeddings") is done via the OpenAI API with "text-embedding-ada-002"')
        st.markdown("#### GPT-3")
        st.write('The chat uses the OpenAI API with the [GPT-3]("https://platform.openai.com/docs/models/gpt-3") model "text-davinci-003""')
        # st.markdown("""---""")
        # st.write('Author: [Daniel Ávila](https://www.linkedin.com/in/daniel-avila-arias/)')
        # st.write('Repo: [Github](https://github.com/davila7/youtube-gpt)')
        # st.write("This software was developed with Code GPT, for more information visit: https://codegpt.co")

    with tab2: 
        st.header("Transcription:")
        if(os.path.exists("audio.mp3")):
            audio_file = open("audio.mp3", 'rb')
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

    with tab3:
        st.header("Summary:")
        if os.path.exists("Summary.txt"):
            with open("Summary.txt", "r") as file:
                transcription_text = file.read()    
                st.write(transcription_text)

    with tab4:
        st.header("Embedding:")
        if os.path.exists("word_embeddings.csv"):
            df = pd.read_csv('word_embeddings.csv')
            st.write(df)

    with tab5:
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
            # st.write(df['embedding'])
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
                max_tokens = 100,
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
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

    with tab6:
        st.header("Sentiment Analysis")
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv('sentiment.csv', sep=',', names=['text','sentiment'])

        # Display the contents of the DataFrame
        st.write(df.head())
        
    with tab7:
        st.header("Whisper transcription")
        # if os.path.exists("whisperoutput.csv"):
        #     df = pd.read_csv('whisperoutput.csv')
        #     st.write(df)
        # if os.path.exists("timestamptranscription.csv"):
        #     df = pd.read_csv('timestamptranscription.csv')
        #     st.write(df)
    
        num_speakers = 2 #@param {type:"integer"}   
        language = 'English' #@param ['any', 'English']
        model_size = 'large' #@param ['tiny', 'base', 'small', 'medium', 'large']

        model_name = model_size
        if language == 'English' and model_size != 'large':
            model_name += '.en'

        path = "audio.mp3"
        if path[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
            path = 'audio.wav'

        model = whisper.load_model(model_size)

        result = model.transcribe(path)
        segments = result["segments"]

        with contextlib.closing(wave.open(path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        audio = Audio()

        def segment_embedding(segment):
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(path, clip)
            return embedding_model(waveform[None])
        
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        embeddings = np.nan_to_num(embeddings)

        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        def time(secs):
            return datetime.timedelta(seconds=round(secs))

        f = open("transcript.txt", "w")

        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')
        f.close()

                    
      
if __name__ == '__main__':
    main()