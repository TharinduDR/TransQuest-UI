import logging

import streamlit as st
from annotated_text import annotated_text

from transquest_ui.app.transquest_wrapper import MicroTransQuestWrapper, MonoTransQuestWrapper

en_de_word = MicroTransQuestWrapper("en_de",  use_cuda=False)
en_de_hter = MonoTransQuestWrapper("en_de_hter", use_cuda=False)
en_de_da = MonoTransQuestWrapper("en_de_da", use_cuda=False)


en_zh_word = MicroTransQuestWrapper("en_zh",  use_cuda=False)
en_zh_hter = MonoTransQuestWrapper("en_zh_hter", use_cuda=False)
en_zh_da = MonoTransQuestWrapper("en_zh_da", use_cuda=False)

multilingual = MicroTransQuestWrapper("en_zh",  use_cuda=False)
multilingual_hter = MonoTransQuestWrapper("en_zh_hter", use_cuda=False)
multilingual_da = MonoTransQuestWrapper("en_zh_hter", use_cuda=False)


def quality_to_rgb(quality: str):
    if quality == "BAD":
        return "rgb(255, 204, 203)"
    else:
        return "rgb(211,211,211)"


def get_model(language: str):

    if language == "en-de":
        return en_de_word, en_de_hter, en_de_da

    elif language == "en-zh":
        return en_zh_word, en_zh_hter, en_zh_da

    elif language == "multilingual":
        return multilingual, multilingual_hter, multilingual_da

    else:
        return None, None, None


def main():
    st.set_page_config(
        page_title='TransQuest UI',
        initial_sidebar_state='expanded',
        layout='wide',
    )

    st.sidebar.title("TransQuest")
    st.sidebar.markdown("Translation Quality Estimation with Cross-lingual Transformers")
    st.sidebar.markdown(
        "[code](https://github.com/TharinduDR/TransQuest)"
    )

    st.sidebar.markdown("---")

    st.sidebar.header("Translation Direction")
    selected_language = st.sidebar.radio(
        'Select the direction of the Translation',
        ["en-de", "en-zh", "multilingual"],
    )

    word_model, hter_model, da_model = get_model(selected_language)

    st.header("Input a Translation")
    st.write(
        "Input a Source and the Target to get the quality from TransQuest"
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        source_text = st.text_area('Source', value="Welcome")

    with col2:
        if selected_language == "en-de":
            target_text = st.text_area('Target', value="Herzlich willkommen")

        elif selected_language == "en-zh":
            target_text = st.text_area('Target', value="欢迎")

        elif selected_language == "multilingual":
            target_text = st.text_area('Target', value="ආයුබෝවන්")

        else:
            target_text = st.text_area('Target', value="Welcome")

    hter_value = hter_model.predict_quality(source_text, target_text)
    da_value = da_model.predict_quality(source_text, target_text)
    source_tags, target_tags = word_model.predict_quality(source_text, target_text)

    hter_value = float(str(hter_value))
    da_value = float(str(da_value))

    hter_value = round(hter_value, 2)
    da_value = round(da_value, 2)

    logging.info(hter_value)
    logging.info(da_value)

    st.header('Translation Quality')

    st.write('Target sentence fixing effort (HTER): ', str(hter_value))
    st.write('Direct Assesement: ', str(da_value))

    for token in target_tags:
        logging.info(token.text)

    source_predictions, target_predictions = st.beta_columns(2)
    with source_predictions:
        source_side = st.beta_container()
    with target_predictions:
        target_side = st.beta_container()

    with source_side:
        text = [
            (token.text, "", quality_to_rgb(token.quality))
            for token in source_tags
            ]
        st.write('Predicted Source Quality (BAD quality words in Red)')
        annotated_text(*text)

    with target_side:
        text = [
            (token.text, "", quality_to_rgb(token.quality))
            for token in target_tags
            ]
        st.write('Predicted Target Quality (BAD quality words and gaps in Red)')
        annotated_text(*text)


if __name__ == "__main__":
    main()




