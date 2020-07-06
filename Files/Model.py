import tensorflow as tf
import argparse
import os

def save_model(model):
    # serialize the model to disk
    print("[INFO] saving mask detector model...")
    model.save('/models/model', save_format="h5")

