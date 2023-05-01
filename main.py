import av
import numpy as np
import matplotlib.pyplot as plt
from gooey import Gooey, GooeyParser
from pathvalidate.argparse import validate_filepath_arg
from skimage.metrics import structural_similarity as ssim


def calculate_frame_similarity(frame_a, frame_b, crop_side=0):
    a = frame_a.to_ndarray(format="rgb24")
    b = frame_b.to_ndarray(format="rgb24")

    if crop_side != 0:
        a = crop_center(a, crop_side)
        b = crop_center(b, crop_side)
    return ssim(a, b, channel_axis=2)


def crop_center(img, crop_side):
    y, x, _ = img.shape
    startx = x // 2 - (crop_side // 2)
    starty = y // 2 - (crop_side // 2)
    return img[starty:starty + crop_side, startx:startx + crop_side, :]


def calculate_ssim_score(video_a_path,
                         video_b_path,
                         n_skip_frames=30,
                         max_frames=None,
                         crop_side=0):
    vid_a_ctr = av.open(video_a_path)
    vid_b_ctr = av.open(video_b_path)

    vid_a_ctr.streams.video[0].thread_type = "AUTO"
    vid_b_ctr.streams.video[0].thread_type = "AUTO"

    vid_a_frame_iter = vid_a_ctr.decode(video=0)
    vid_b_frame_iter = vid_b_ctr.decode(video=0)

    processed_frames = []
    ssim_results = []
    for idx, (frame_a,
              frame_b) in enumerate(zip(vid_a_frame_iter, vid_b_frame_iter)):
      
        if max_frames is not None and len(processed_frames) > max_frames:
            break
        if idx % n_skip_frames != 0:
            continue
        
        ssim_results.append(
            calculate_frame_similarity(frame_a, frame_b, crop_side))
        processed_frames.append(idx)


    vid_a_frame_iter = np.array(processed_frames)
    ssim_results = np.array(ssim_results)

    vid_a_ctr.close()
    vid_b_ctr.close()

    return processed_frames, ssim_results


def parse_cli():
    parser = GooeyParser()
    parser.add_argument(
        "source_path",
        type=validate_filepath_arg,
        help="Path to source video file",
        widget="FileChooser",
    )
    parser.add_argument(
        "encode_path",
        type=validate_filepath_arg,
        help="Path to encoded video file",
        widget="FileChooser",
    )
    parser.add_argument(
        "-n",
        "--num-skip-frames",
        type=int,
        default=15,
        help="Step between frames for which SSIM is evaluated. [Default: 15]",
    )
    parser.add_argument(
        "-m",
        "--max_num_frames",
        type=int,
        default=500,
        help="Maximum number of frames to be processed. [Default: 500]",
    )
    parser.add_argument(
        "-c",
        "--crop_side",
        type=int,
        default=0,
        help=
        "Crop a center square from the video with the specified side. 0 means don't crop. [Default: 0]",
    )
    
    options = parser.parse_args()
    return options


@Gooey(
    program_name="Source-encode SSIM calculator",
    program_description=
    "Calculates the SSIM score frame by frame between a source and encode.\nRequires both to have the same resolution and number of frames",
    default_size=(610, 610),
)
def main():
    cli = parse_cli()
    plt.figure(f"SSIM Score for {cli.source_path}")
    plt.xlabel("Frame number")
    plt.ylabel("SSIM Score")
    frames, scores = calculate_ssim_score(
        cli.source_path,
        cli.encode_path,
        n_skip_frames=cli.num_skip_frames,
        max_frames=cli.max_num_frames + 1,
        crop_side=cli.crop_side,
    )
    avg_score = np.average(scores)
    plt.plot(frames, scores, label="Current SSIM score")
    plt.hlines(
        [avg_score],
        xmin=frames[0],
        xmax=frames[-1],
        colors=["red"],
        label=f"Average SSIM score: {avg_score:.3f}",
    )
    print(f"Average SSIM score for whole video: {avg_score}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
