import os
import subprocess
import imageio
import shutil

def concatenate_videos_ffmpeg(output_video):
    # Create a file list for ffmpeg (filenames only)
    file_list_path = 'file_list.txt'
    with open(file_list_path, 'w') as f:
        for video_file in sorted(os.listdir('.')):
            if video_file.endswith('.mp4') and not video_file.startswith('lunar_lander_sac_combined'):
                f.write(f"file '{video_file}'\n")
    # Run ffmpeg from the current directory
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list_path,
        '-c', 'copy', output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"Combined video saved as {output_video}")

def convert_video_to_gif(input_video, output_gif):
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']
    frames = [frame for frame in reader]
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF saved as {output_gif}")

def main():
    output_video = 'lunar_lander_sac_combined.mp4'
    output_gif = 'lunar_lander_sac_combined.gif'
    concatenate_videos_ffmpeg(output_video)
    convert_video_to_gif(output_video, output_gif)

if __name__ == "__main__":
    main() 