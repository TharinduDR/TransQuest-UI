import logging

import streamlit as st
from annotated_text import annotated_text
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

from transquest_ui.app.args import microtransquest_config, monotransquest_config


class PredictedToken:
    def __init__(self, text, quality):
        self.text = text
        self.quality = quality


model_args = {"use_multiprocessing": False}

en_de_word = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", args=microtransquest_config, labels=["OK", "BAD"], use_cuda=False)
en_de_da = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-en_de-wiki", args=monotransquest_config, num_labels=1, use_cuda=False)


# en_zh_word = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_zh-wiki", args=model_args, labels=["OK", "BAD"], use_cuda=False)
# en_zh_da = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-en_zh-wiki", args=model_args, num_labels=1, use_cuda=False)
#
# multilingual = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_zh-wiki", args=model_args, labels=["OK", "BAD"], use_cuda=False)
# multilingual_da = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-multilingual", args=model_args, num_labels=1, use_cuda=False)

logging.info("Finished loading models")


def quality_to_rgb(quality: str):
    if quality == "BAD":
        return "rgb(255, 0, 0)"
    else:
        return "rgb(211,211,211)"


def get_model(language: str):

    if language == "en-de":
        return en_de_word, en_de_da

    # elif language == "en-zh":
    #     return en_zh_word, en_zh_da
    #
    # elif language == "multilingual":
    #     return multilingual, multilingual_da

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

    word_model, da_model = get_model(selected_language)

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

    da_value, raw_outputs = da_model.predict([[source_text, target_text]])
    source_tags, target_tags = word_model.predict([[source_text, target_text]])

    source_words = source_text.split()
    target_words = target_text.split()

    source_predicted_tokens = []
    target_predicted_tokens = []
    for source_word, source_tag in zip(source_words, source_tags[0]):
        source_predicted_token = PredictedToken(source_word, source_tag)
        source_predicted_tokens.append(source_predicted_token)
    gap_index = 0
    word_index = 0
    for prediction_id, prediction in enumerate(target_tags[0]):
        if prediction_id % 2 == 0:
            target_predicted_token = PredictedToken("<GAP>", prediction)
            gap_index += 1
            target_predicted_tokens.append(target_predicted_token)
        else:
            target_predicted_token = PredictedToken(target_words[word_index], prediction)
            word_index += 1
            target_predicted_tokens.append(target_predicted_token)

    da_value = float(str(da_value))

    da_value = round(da_value, 2)

    logging.info(da_value)

    st.header('Translation Quality')

    st.write('Direct Assesement: ', str(da_value))

    source_predictions, target_predictions = st.beta_columns(2)
    with source_predictions:
        source_side = st.beta_container()
    with target_predictions:
        target_side = st.beta_container()

    with source_side:
        text = [
            (token.text, "", quality_to_rgb(token.quality))
            for token in source_predicted_tokens
            ]
        st.write('Predicted Source Quality (BAD quality words in Red)')
        annotated_text(*text)

    with target_side:
        text = [
            (token.text, "", quality_to_rgb(token.quality))
            for token in target_predicted_tokens
            ]
        st.write('Predicted Target Quality (BAD quality words and gaps in Red)')
        annotated_text(*text)


if __name__ == "__main__":
    main()




