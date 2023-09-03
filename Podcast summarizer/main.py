from api_communication import *
import streamlit as st
import json

st.title('Welcome to my application that creates podcast summaries')
episode_id = st.sidebar.text_input('Please input an episode id')
button = st.sidebar.button('Get podcast summary!', on_click = save_transcript, args = (episode_id,))

   

if button:
    filename = episode_id + '_chapters.json'
    with open (filename, 'r') as f:
        data = json.load(f)

        chapters = data['chapters']
        podcast_title = data['podcast_title'] 
        episode_title = data['episode_title']
        thumbnail = data['episode_thumbnail']
        
    st.header(f'{podcast_title} - {episode_title}')
    st.image(thumbnail)
    for chp in chapters:
        with st.expander(chp['gist']):
            chp['summary']    


