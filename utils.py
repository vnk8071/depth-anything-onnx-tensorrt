import numpy as np
import cv2
from matplotlib import cm
import argparse
import os
import platform
import pycuda.driver as cuda
import tensorrt as trt
import logging


logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger()
TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class HostDeviceMem:
    """
    Host and Device Memory Package
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class DPTTrt:
    """Depth Anything v2 inference class"""

    def __init__(self, args) -> None:
        self.reshape_size = [518, 518]
        self.model_path = args.model_path
        self.__trt_init__(
            self.model_path,
            dynamic_shape=False,
            batch_size=1,
        )

    def __trt_init__(self, trt_file=None, dynamic_shape=False, gpu_idx=0, batch_size=1):
        """
        Init tensorrt.
        :param trt_file:    tensorrt file.
        :return:
        """
        cuda.init()
        self._batch_size = batch_size
        self._device_ctx = cuda.Device(gpu_idx).make_context()
        self._engine = self._load_engine(trt_file)
        self._context = self._engine.create_execution_context()
        if not dynamic_shape:
            (
                self._input,
                self._output,
                self._bindings,
                self._stream,
            ) = self._allocate_buffers(self._context)

        logger.info("Dpt model <loaded>...")

    def _load_engine(self, trt_file):
        """
        Load tensorrt engine.
        :param trt_file:    tensorrt file.
        :return:
            ICudaEngine
        """
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self, context):
        """
        Allocate device memory space for data.
        :param context:
        :return:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self._engine:
            size = (
                trt.volume(self._engine.get_binding_shape(binding))
                * self._engine.max_batch_size
            )
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_infer(self, data):
        """
        Real inference process.
        :param model:   Model objects
        :param data:    Preprocessed data
        :return:
            output
        """
        # Copy data to input memory buffer
        [np.copyto(_inp.host, data.ravel()) for _inp in self._input]
        # Push to device
        self._device_ctx.push()
        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self._input.device, self._input.host, self._stream)
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self._stream)
            for inp in self._input
        ]
        # Run inference.
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )
        # Transfer predictions back from the GPU.
        # cuda.memcpy_dtoh_async(self._output.host, self._output.device, self._stream)
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self._stream)
            for out in self._output
        ]
        # Synchronize the stream
        self._stream.synchronize()
        # Pop the device
        self._device_ctx.pop()

        return [out.host.reshape(self._batch_size, -1) for out in self._output[::-1]]

    def inference(self, input_frame_array) -> np.ndarray:
        """Inference core
        :param input_frame_array:   numpy.ndarray
        :return:
            net_out
        """
        trt_inputs = [input_frame_array]
        trt_inputs = np.vstack(trt_inputs)
        result = self.trt_infer(trt_inputs)
        net_out = result[0].reshape(self.reshape_size[0], self.reshape_size[1])
        return net_out

    def __del__(self):
        self._device_ctx.pop()


def preprocess(frame, input_size):
    """Preprocess the frame for model inference."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)[None].astype("float32")
    return image


def postprocess(
    shape_info: tuple, depth: np.ndarray, grayscale, crop_region=None
) -> np.ndarray:
    """Postprocess core
    :param shape_info:   tuple
    :param depth:        numpy.ndarray
    :return:
        net_out
    """
    cmap = cm.get_cmap("Spectral_r")
    if crop_region is not None:
        x, y, w, h = crop_region.split(" ")
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Resize crop_region to the depth map size
        x_scale, y_scale, w_scale, h_scale = (
            int(x / shape_info[0] * depth.shape[0]),
            int(y / shape_info[1] * depth.shape[1]),
            int(w / shape_info[0] * depth.shape[0]),
            int(h / shape_info[1] * depth.shape[1]),
        )
        crop_depth = depth[y_scale : y_scale + h_scale, x_scale : x_scale + w_scale]
        depth = (
            (crop_depth - crop_depth.min())
            / (crop_depth.max() - crop_depth.min())
            * 255.0
        )
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (w, h))
    else:
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (shape_info[0], shape_info[1]))

    if grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth
