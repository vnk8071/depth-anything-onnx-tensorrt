import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import cv2
import numpy as np
import argparse
import os
from matplotlib import cm

class DepthAnythingTriton:
    def __init__(self, model_name, model_version=1, client_type="grpc", server_url="192.168.81.232"):
        self.model_name = model_name
        self.model_version = model_version
        self.client_type = client_type
        self.server_url = server_url

        if client_type == "grpc":
            self.client = grpcclient.InferenceServerClient(url=f"{server_url}:8001", verbose=False)
        else:
            self.client = httpclient.InferenceServerClient(url=f"{server_url}:8000", verbose=False)
        
        assert self.client.is_server_ready()

        if client_type == "grpc":
            self.input_metadata = self.client.get_model_metadata(model_name).inputs[0]
            self.input_h, self.input_w = self.input_metadata.shape[-2:]
        else:
            self.input_metadata = self.client.get_model_metadata(model_name)["inputs"][0]
            self.input_h, self.input_w = self.input_metadata["shape"][-2:]

        self.output_name = "output"
        self.cmap = cm.get_cmap('Spectral_r')

    def preprocess(self, frame):
        image = frame
        
        if self.client_type == "grpc":
            if self.model_name == "depth_anything_dynamic":
                inps = [grpcclient.InferInput(self.input_metadata.name, image.shape, self.input_metadata.datatype)]
            elif self.model_name == "depth_anything":
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
                image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                image = image.transpose(2, 0, 1).astype("float32")
                image = np.expand_dims(image, axis=0)
                inps = [grpcclient.InferInput(self.input_metadata.name, self.input_metadata.shape, self.input_metadata.datatype)]
            inps[0].set_data_from_numpy(image)
            outputs_names = [self.output_name]
            outs = [grpcclient.InferRequestedOutput(name) for name in outputs_names]
        else:
            if self.model_name == "depth_anything_dynamic":
                inps = [httpclient.InferInput(self.input_metadata["name"], image.shape, self.input_metadata["datatype"])]
            elif self.model_name == "depth_anything":
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
                image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                image = image.transpose(2, 0, 1).astype("float32")
                image = np.expand_dims(image, axis=0)
                inps = [httpclient.InferInput(self.input_metadata["name"], self.input_metadata["shape"], self.input_metadata["datatype"])]
            inps[0].set_data_from_numpy(image)
            outputs_names = [self.output_name]
            outs = [httpclient.InferRequestedOutput(name) for name in outputs_names]
        return inps, outs

    def infer(self, inps, outs):
        print("model_name: ", self.model_name)
        print("model_version: ", self.model_version)
        results = self.client.infer(model_name=self.model_name, model_version=self.model_version, inputs=inps, outputs=outs)
        depth_pred = results.as_numpy(self.output_name).squeeze()
        return depth_pred

    def postprocess(self, shape_info, depth, grayscale, crop_region=None):
        if crop_region is not None:
            x, y, w, h = crop_region.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            x_scale, y_scale, w_scale, h_scale = int(x / shape_info[1] * depth.shape[0]), int(y / shape_info[0] * depth.shape[1]), int(w / shape_info[1] * depth.shape[0]), int(h / shape_info[0] * depth.shape[1])
            crop_depth = depth[y_scale:y_scale+h_scale, x_scale:x_scale+w_scale]
            crop_depth = cv2.resize(crop_depth, (w, h))
            depth = (crop_depth - crop_depth.min()) / (crop_depth.max() - crop_depth.min()) * 255.0
            depth = depth.astype(np.uint8)
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.resize(depth, (shape_info[1], shape_info[0]))
        
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        return depth

    def remove_outliers(self, arr):
        mean = arr.mean()
        std = np.std(arr)
        min_val = mean - std * 2
        max_val = mean + std * 2
        return np.clip(arr, min_val, max_val)

    def predict(self, frame, grayscale=False, crop_region=None):
        if isinstance(frame, str):
            image = cv2.imread(frame).astype(np.float32)
        else:
            image = frame.astype(np.float32)
        inps, outs = self.preprocess(image)
        depth_pred = self.infer(inps, outs)
        depth_pred = self.remove_outliers(depth_pred)
        depth_pred = self.postprocess(image.shape[:2], depth_pred, grayscale, crop_region)
        if crop_region is not None:
            raw_frame_copy = frame.copy()
            x, y, w, h = crop_region.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            raw_frame_copy[y : y + h, x : x + w, :] = depth_pred
            return raw_frame_copy
        else:
            return depth_pred

    def predict_video(self, video_path, output_path, grayscale=False, crop_region=None):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            depth_pred = self.predict(frame, grayscale, crop_region)
            out.write(depth_pred)

        cap.release()
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything Inference")
    parser.add_argument("--model_name", type=str, default="depth_anything", help="Name of the model")
    parser.add_argument("--model_version", type=str, default="1", help="Version of the model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or video")
    parser.add_argument("--output_dir", type=str, default="output_triton", help="Path to save the output image or video")
    parser.add_argument("--is_video", action="store_true", help="Specify if the input is a video")
    parser.add_argument("--grayscale", action="store_true", help="Save depth map in grayscale")
    parser.add_argument("--crop_region", type=str, default=None, help="Crop region for the depth map")
    parser.add_argument("--client_type", type=str, choices=["grpc", "http"], default="grpc", help="Type of Triton client to use")

    args = parser.parse_args()

    depth_anything = DepthAnythingTriton(args.model_name, args.model_version, args.client_type)
    filename = args.input_path.split("/")[-1]
    output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    if args.is_video:
        depth_anything.predict_video(args.input_path, output_path, args.grayscale, args.crop_region)
    else:
        depth_pred = depth_anything.predict(args.input_path, args.grayscale, args.crop_region)
        cv2.imwrite(output_path, depth_pred)
    print("Save output path: ", output_path)
