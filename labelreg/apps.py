"""
This is a tutorial code. Standalone applications using TensorFlow may be slow due to data setup and transfer.
Whenever possible, use the same tf session for model restoring, inference and testing in an application.
"""
import tensorflow as tf

class ImageMinibatch:
    def __init__(self, imaged, ddf):