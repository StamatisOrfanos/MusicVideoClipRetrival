import ffmpeg
import os, shutil
import whisper
import torch



def data_creation(demo_data, video_path):
    create_frames(demo_data, video_path)
    create_lyrics(demo_data, video_path)


def create_frames(demo_data, video_path):

    os.makedirs(demo_data, exist_ok=True)
    os.makedirs(os.path.join(demo_data, 'frames'))  

    input_file = ffmpeg.input(video_path)
    output_file_pattern = os.path.join(os.path.join(demo_data, 'frames'), 'frame_%d.png')

    # Check how long is a video to take a custom frequency for each video
    probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    duration = float(video_info['duration'])

    if (duration >= 300):
        fps = 'fps=0.5'
    elif (duration < 300 and duration >= 200):
        fps = 'fps=1'
    else:
        fps = 'fps=2'

    output = ffmpeg.output(input_file, output_file_pattern, vf=fps, start_number=0)
    ffmpeg.run(output)


def create_lyrics(demo_data, video_path):
    os.makedirs(demo_data, exist_ok=True)
    model = whisper.load_model("base.en")

    video_name = 'demo'
    video_lyrics = model.transcribe(video_path)


    # Produce the text file with the lyrics
    file_path = demo_data + '/{}.txt'.format(video_name)
    file = open(file_path, 'a')

    for i, seg in enumerate(video_lyrics['segments']):
        file.write(seg['text'] + '\n')

    file.close()


# Define cosine similarity function
def cosine_similarity(test_image_features, train_data_features, dim=1, eps=1e-8):
    dot_product = torch.sum(test_image_features * train_data_features, dim=dim)
    norm_test = torch.norm(test_image_features, dim=dim)
    norm_data = torch.norm(train_data_features, dim=dim)
    return dot_product / (norm_test * norm_data).clamp(min=eps)