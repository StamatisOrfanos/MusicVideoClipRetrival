# MusicVideoClipRetrival
This is a project for my Masters degree, for the Multimodal master degree course


In order to run run the application the following steps are necessary:
1. Include the data folder in the same directory (90 most popular songs in [Spotify](https://open.spotify.com/)
2. Install the libraries in the requirements.txt
3. To set up the image data for the Resnet-50 model run the frames_set_up.ipynb to create the frames for each music video clip.
4. To set up the text data for the Bert-uncased model run the lyrics_set_up.ipynb to create both text files that include the lyrics and csv files that include vital information for the content of the music video clips.
5. Run the resnet_model.ipynb, that will create the trained torch model.
6. Run the bert_base_model to get the results of the textual model.
