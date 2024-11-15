import cv2
import glob
from matplotlib import cm
import numpy as np
import os
import gradio as gr
import shutil
from utils import DPTTrt

def preprocess(frame, input_size):
    """Preprocess the frame for model inference."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)[None].astype("float32")
    return image


def postprocess(shape_info: tuple, depth: np.ndarray, crop_region=None) -> np.ndarray:
    """Postprocess core
    :param shape_info:   tuple
    :param depth:        numpy.ndarray
    :return:
        net_out
    """
    cmap = cm.get_cmap('Spectral_r')
    if crop_region != "":
        x, y, w, h = map(int, crop_region.split(" "))

        # Resize crop_region to the depth map size
        x_scale, y_scale, w_scale, h_scale = int(x / shape_info[0] * depth.shape[0]), int(y / shape_info[1] * depth.shape[1]), int(w / shape_info[0] * depth.shape[0]), int(h / shape_info[1] * depth.shape[1])
        crop_depth = depth[y_scale:y_scale+h_scale, x_scale:x_scale+w_scale]
        crop_depth = cv2.resize(crop_depth, (w, h))

        depth = (crop_depth - crop_depth.min()) / (crop_depth.max() - crop_depth.min()) * 255.0
        depth = depth.astype(np.uint8)
    else:
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (shape_info[0], shape_info[1]))
    depth_gray = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return depth_gray, depth

def crop_zone(filename, crop_region):
    video = cv2.VideoCapture(filename)
    frame = video.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if crop_region == "":
        return frame
    crop_coords = crop_region.split(" ")
    if len(crop_coords) != 4:
        raise gr.Error("Crop region must be in the format 'x y w h'", duration=5)
    x, y, w, h = map(int, crop_region.split(" "))
    crop = frame[y:y+h, x:x+w]
    return crop

def get_infer_gradio(model, filename, crop_region=None):
    class Args:
        def __init__(self, model_path, crop_region, mode):
            self.model_path = model_path
            self.input_size = 518
            self.crop_region = crop_region
            self.mode = mode
            self.filename = filename

    if model == "small":
        model_path = "models/depth_anything_v2_vits.engine"
    elif model == "base":
        model_path = "models/depth_anything_v2_vitb.engine"
    else:
        raise ValueError("Unsupported model type. Choose either 'small' or 'base'.")

    args = Args(model_path, crop_region, "trt")
    return process_video(DPTTrt(args), args, filename)


with gr.Blocks() as iface:
    gr.Markdown("# Depth Anything V2 Inference")
    gr.Markdown("Upload video and select the appropriate options to run inference.")

    with gr.Row():
        input_file = gr.Video(label="Input Video")
        image_crop = gr.Image(label="Crop zone")
        depth_file = gr.Video(label="Depth Video", autoplay=True)
    crop_region = gr.Textbox(value="", label="Crop Region (x y w h). If empty, the entire frame will be used.")
    with gr.Row():
        crop_button = gr.Button("Preview Crop Zone")
        submit_button = gr.Button("Predict Depth")
    model = gr.Radio(choices=["small", "base"], label="Model", value="small")
    with gr.Row():
        raw_depth_file = gr.Video(label="Raw depth map with thresh birany", autoplay=True)
        depth_file_crop = gr.Video(label="Depth crop zone", autoplay=True)
        grayscale_depth_file = gr.Video(label="Grayscale depth map", autoplay=True)

    def process_video(session, args, filename):
        print(f"Progress: {filename}")
        shutil.copy(filename, "input/")
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        x, y, w, h = map(int, args.crop_region.split(" ")) if args.crop_region != "" else (0, 0, frame_width, frame_height)
        output_depth_path = os.path.join(
            "output", "depth_" + os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out_depth_crop = cv2.VideoWriter(
            output_depth_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (w, h),
        )
        output_path = os.path.join(
            "output", os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (frame_width, frame_height),
        )
        output_raw_path = os.path.join(
            "output", "raw_" + os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out_raw = cv2.VideoWriter(
            output_raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (518, 518),
        )

        output_grayscale_path = os.path.join(
            "output", "grayscale_" + os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out_grayscale = cv2.VideoWriter(
            output_grayscale_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (frame_width, frame_height),
        )

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            image = preprocess(raw_frame, args.input_size)
            depth = session.inference(image)
            depth_normalized = depth.astype(np.uint8)
            depth_normalized = cv2.threshold(depth_normalized, 0, 255, cv2.THRESH_BINARY)[1]
            depth_normalized = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
            out_raw.write(depth_normalized)

            depth_grayscale, depth = postprocess(
                (frame_width, frame_height), depth, args.crop_region
            )
            out_depth_crop.write(depth)

            if args.crop_region != "":
                raw_frame_copy = raw_frame.copy()
                raw_frame_copy[y : y + h, x : x + w, :] = depth_grayscale
                out_grayscale.write(raw_frame_copy)
                raw_frame_copy[y : y + h, x : x + w, :] = depth
                out.write(raw_frame_copy)
            else:
                out_grayscale.write(depth_grayscale)
                out.write(depth)

        raw_video.release()
        out_depth_crop.release()
        out.release()
        out_raw.release()
        out_grayscale.release()
        return [output_path, output_raw_path, output_depth_path, output_grayscale_path]

    crop_button.click(
        crop_zone, inputs=[input_file, crop_region], outputs=[image_crop]
    )
    submit_button.click(
        get_infer_gradio, inputs=[model, input_file, crop_region], outputs=[depth_file, raw_depth_file, depth_file_crop, grayscale_depth_file]
    )

    example_files = glob.glob("assets/*mp4")
    example_files = [["small", f, 518] for f in example_files if f.endswith(".mp4")]
    examples = gr.Examples(example_files, label="Example Videos", inputs=[model, input_file], outputs=[depth_file, raw_depth_file, depth_file_crop, grayscale_depth_file], fn=get_infer_gradio)


if __name__ == '__main__':
    iface.queue().launch(server_name="0.0.0.0", server_port=7860)
