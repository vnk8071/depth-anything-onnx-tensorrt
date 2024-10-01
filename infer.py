import argparse
import cv2
from enum import auto
import glob
from matplotlib import cm
import numpy as np
import os
import onnxruntime as ort
import tensorrt as trt
from utils import DPTTrt, preprocess, postprocess


class InferenceDevice:
    cpu = auto()
    cuda = auto()


def create_onnx_session(model_path, device):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = False
    providers = ["CPUExecutionProvider"]
    if device == InferenceDevice.cuda:
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )
    return session


def process_video(session, args, filenames):
    for k, filename in enumerate(filenames):
        print(f"Progress {k+1}/{len(filenames)}: {filename}")

        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(
            args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".mp4"
        )
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (frame_width, frame_height),
        )

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            image = preprocess(raw_frame, args.input_size)
            if args.mode == "onnx":
                ort_input = session.get_inputs()[0].name
                ort_output = session.get_outputs()[0].name
                depth = session.run([ort_output], {ort_input: image})[0][0, :, :]
            elif args.mode == "trt":
                depth = session.inference(image)

            depth = postprocess(
                (frame_width, frame_height), depth, args.grayscale, args.crop_region
            )

            if args.crop_region is not None:
                raw_frame_copy = raw_frame.copy()
                x, y, w, h = args.crop_region.split(" ")
                x, y, w, h = int(x), int(y), int(w), int(h)
                raw_frame_copy[y : y + h, x : x + w, :] = depth
                out.write(raw_frame_copy)
            else:
                out.write(depth)

        raw_video.release()
        out.release()


def process_images(session, args, filenames):
    for k, filename in enumerate(filenames):
        print(f"Processing {k+1}/{len(filenames)}: {filename}")

        raw_image = cv2.imread(filename)
        image = preprocess(raw_image, args.input_size)

        if args.mode == "onnx":
            ort_input = session.get_inputs()[0].name
            ort_output = session.get_outputs()[0].name
            depth = session.run([ort_output], {ort_input: image})[0][0, 0, :, :]
        elif args.mode == "trt":
            depth = session.inference(image)

        depth = postprocess(
            (raw_image.shape[1], raw_image.shape[0]),
            depth,
            args.grayscale,
            args.crop_region,
        )

        if args.crop_region is not None:
            raw_image_copy = raw_image.copy()
            x, y, w, h = args.crop_region.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            raw_image_copy[y : y + h, x : x + w, :] = depth
            output_image = raw_image_copy
        else:
            output_image = depth

        output_path = os.path.join(args.outdir, os.path.basename(filename))
        cv2.imwrite(output_path, output_image)


def main(args):
    if args.mode == "onnx":
        session = create_onnx_session(args.model_path, args.device)
    elif args.mode == "trt":
        session = DPTTrt(args)
    else:
        raise ValueError("Unsupported mode. Choose either 'onnx' or 'trt'.")

    if os.path.isfile(args.input_path):
        if args.input_path.endswith("txt"):
            with open(args.input_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.input_path]
    else:
        filenames = glob.glob(os.path.join(args.input_path, "**/*"), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    if args.input_type == "video":
        process_video(session, args, filenames)
    elif args.input_type == "image":
        process_images(session, args, filenames)
    else:
        raise ValueError("Unsupported input type. Choose either 'video' or 'image'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input video/image or directory containing videos/images.",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        required=True,
        choices=["video", "image"],
        help="Input type: video or image.",
    )
    parser.add_argument(
        "--input_size", type=int, default=518, help="Input size for the model."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./vis_output",
        help="Output directory for the processed videos/images.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder type for the model.",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="Do not apply colorful palette.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/depth_anything_v2_vits.onnx",
        help="Path to model.",
    )
    parser.add_argument(
        "--device",
        type=InferenceDevice,
        default=InferenceDevice.cpu,
        help="Inference device.",
    )
    parser.add_argument(
        "--crop-region",
        type=str,
        default=None,
        help="Get value x,y,w,h with space i.e. 0 0 500 500",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["onnx", "trt"],
        help="Inference mode: onnx or trt",
    )

    args = parser.parse_args()
    main(args)
