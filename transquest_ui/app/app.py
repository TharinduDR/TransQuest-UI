import streamlit as st
from annotated_text import annotated_text

from transquest_ui.app.transquest_wrapper import MicroTransQuestWrapper, MonoTransQuestWrapper

en_de_word = MicroTransQuestWrapper("en_de",  use_cuda=False)
en_de_hter = MonoTransQuestWrapper("en_de_hter", use_cuda=False)
en_de_da = MonoTransQuestWrapper("en_de_da", use_cuda=False)


def quality_to_rgb(quality: str):
    if quality == "BAD":
        return "rgb(255, 0, 0)"
    else:
        return "rgb(211,211,211)"


def get_model(language: str):

    if language == "en-de":
        return en_de_word, en_de_hter, en_de_da

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
        target_text = st.text_area('Target', value="Herzlich willkommen")

    hter_value = hter_model.predict_quality(source_text, target_text)
    da_value = da_model.predict_quality(source_text, target_text)
    source_tags, target_tags = word_model.predict(source_text, target_text)

    st.header('Translation Quality')

    st.write('Target sentence fixing effort (HTER): ', hter_value)
    st.write('Direct Assesement: ', da_value)

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
        st.write('Predicted Source Tags')
        annotated_text(*text)

    with target_side:
        text = [
            (token.text, "", quality_to_rgb(token.quality))
            for token in target_tags
            ]
        st.write('Predicted Target Tags')
        annotated_text(*text)


if __name__ == "__main__":
    main()




